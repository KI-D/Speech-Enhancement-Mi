import sys
import os
sys.path.append("../")
import torch
from torch import nn
import numpy as np
import argparse
import yaml
import json
import os
from torch.cuda.amp import autocast as autocast
from torch.cuda.amp import GradScaler as GradScaler
from torch.utils.tensorboard import SummaryWriter
from data_c import LibriPartyDataset
from torch.utils.data import DataLoader
from utility import *
import torch.distributed as dist
from speechbrain.nnet.schedulers import NoamScheduler
from hifigan import Hifi_GAN


DEBUG = True
class Processor(object):
    def __init__(self, args):
        global DEBUG
        with open(args.config_path,'r',encoding='utf-8') as f:
            config = f.read()
        config = yaml.load(config, Loader=yaml.FullLoader)
        DEBUG = config['config']['DEBUG']
        config['config']['gpu'] = args.gpu
        config['user_defined_name'] = args.user_defined_name
        config['config']['local_rank'] = args.local_rank
        self.config = config
        self.stage = 1
        self.dataset = LibriPartyDataset()
        self.init_seed(config['config']['seed'])
        self.init_modules()

        self.epoch = -1
        self.train_step = 0
        self.dev_step = 0
        self.last_loss = 1e8
        self.g_optimizer = None
        self.d_optimizer = None
        self.g_scheduler = None
        self.d_scheduler = None


    def init_seed(self, seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        torch.backends.cudnn.deterministic = True

    def init_modules(self):
        module_names = ["Hifi_GAN"]
        for name in module_names:
            setattr(self, name, globals()[name](**self.config[name]))
    
    def set_writter(self, stage_name):
        log_dir = self.config['config']['log_dir']
        log_module = os.path.join(log_dir, stage_name, self.config['user_defined_name'])
        if not os.path.exists(log_module):
            os.makedirs(log_module)
        self.writer = SummaryWriter(log_module)
    
    def set_device(self, module_names, frozen_names, devices):
        # gpus = [str(x) for x in self.config['config']['gpu']]
        # os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(gpus)
        for name,device in zip(module_names, devices):
            if device >= 0:
                getattr(self, name).to("cuda:"+str(device))
            if name in frozen_names:
                for param in getattr(self, name).parameters():
                    param.requires_grad = False
    
    def save_modules(self, stage_name, module_names, optimizer_names, scheduler_names):
        check_dir = self.config['config']['checkpoint_dir']
        stage_dir = os.path.join(check_dir, stage_name, self.config['user_defined_name'])
        if not os.path.exists(stage_dir):
            os.makedirs(stage_dir)
        
        for name in module_names:
            path = os.path.join(stage_dir, name+'.pth')
            torch.save(getattr(self, name).state_dict(), path)
        for name in optimizer_names+scheduler_names:
            path = os.path.join(stage_dir, name+'.pth')
            if self.stage == 3 and name.split('_')[1] == 'scheduler':
                getattr(self, name).save(path)
            else:
                torch.save(getattr(self, name).state_dict(), path)
        path = os.path.join(stage_dir, 'Epoch.pth')
        torch.save({"Epoch": self.epoch, "Train_Step":self.train_step, "Dev_Step":self.dev_step, "Last_Loss":self.last_loss}, path)
    

    def load_modules(self, stage_name, module_names, optimizer_names, scheduler_names, load_model=False):
        if stage_name == "fronzen":
            stage_dir = "fronzen_modules"
        else:
            if load_model:
                stage_name = "stage" + str(self.stage-1)
                print(f"Load Before Model: {stage_name}")
            stage_dir = os.path.join(self.config['config']['checkpoint_dir'], stage_name, self.config['user_defined_name'])
        
        for name in module_names:
            path = os.path.join(stage_dir, name+'.pth')
            getattr(self, name).load_state_dict(torch.load(path), strict=False)
        
        if not load_model:
            for name in optimizer_names+scheduler_names:
                path = os.path.join(stage_dir, name+'.pth')
                if self.stage == 3 and name.split('_')[1] == 'scheduler':
                    getattr(self, name).load(path)
                else:
                    getattr(self, name).load_state_dict(torch.load(path))
            path = os.path.join(stage_dir, 'Epoch.pth')
            param = torch.load(path)
            self.epoch = param['Epoch']
            self.train_step = param['Train_Step']
            self.dev_step = param['Dev_Step']
            self.last_loss = param['Last_Loss']


    def train(self, stage, resume = False, load_model = False):
        self.stage = stage
        stagename = "stage" + str(stage)
        num_epoch = self.config[stagename]['num_epoch']
        optimizer_names = ["g_optimizer"]
        scheduler_names = ["g_scheduler"]
        if stage == 3:
            optimizer_names += ["d_optimizer"]
            scheduler_names += ["d_scheduler"]
        
        model, scaler = self.set_denoise(stage, resume, load_model)
        self.set_writter(stagename)
        history = {"train_step":self.train_step, "dev_step":self.dev_step, "epoch":self.epoch, "train_loss":0.0, "train_g":0.0, "train_d":0.0, "dev_last_loss":self.last_loss, "dev_loss":0.0, "dev_g":0.0, "dev_d":0.0}
        for epoch in range(self.epoch+1, num_epoch):
            history["epoch"] = epoch
            model.train()
            history = self.denoise(model, scaler, stage, "train", history)
            self.train_step = history["train_step"]
            model.eval()
            history = self.denoise(model, scaler, stage, "dev", history)
            self.epoch = epoch
            self.dev_step = history["dev_step"]
            self.last_loss = history["dev_last_loss"]
            self.save_modules(stagename, [], optimizer_names, scheduler_names)
        


    def denoise(self, model, scaler, stage, mode, history):
        epoch = self.epoch
        stagename = "stage" + str(stage)
        config = self.config[stagename]
        max_grad_norm = self.config['config']['max_grad_norm']
        avg_step = self.config['config']['avg_step']
        batch_size = config['batch_size']
        num_workers = config['num_workers']
        num_mic = config['num_mic']
        gradient_accumulation = config['gradient_accumulation']
        augment = False if stage == 1 else True

        self.dataset.set_attribute(mode, augment = augment)
        self.dataset.init_seed(epoch+1)
        self.dataset.buffer = []
        # ################################################################################
        # sampler = torch.utils.data.distributed.DistributedSampler(self.dataset)
        # dataloader = DataLoader(self.dataset, batch_size = batch_size, shuffle=True, collate_fn=collate_fn, num_workers=num_workers, pin_memory = True, sampler=sampler)
        # ################################################################################
        dataloader = DataLoader(self.dataset, batch_size = batch_size, shuffle=False, num_workers=num_workers)
        step_name = mode + '_step'
        if stage == 3:
            loss_names = [mode + '_d', mode + '_g']
        else:
            loss_names = [mode + '_loss']
        
        global_step = 0
        for data in dataloader:
            device = 'cuda:1'
            mixture = data['mix'].to(device)
            source = data['source'].to(device).squeeze(1)
            noise = data['noise'].to(device)
            length = data['length'].to(device)
            num_mics = torch.tensor([num_mic]*batch_size).long().to(device)
            source = source[:,0].unsqueeze(1)
            ##########################################################################
            # mixture = data['mix'].cuda()
            # source = data['source'].cuda()
            # noise = data['noise'].cuda()
            # length = data['length'].cuda()
            # spk = data['spk'].cuda()
            # num_mics = torch.tensor([num_mic]*batch_size).long().cuda()
            ##########################################################################
            reset = not data['flag']
            if mode == 'train':
                if stage < 3:
                    loss = model.train_stage(mixture, source, stage, reset = reset)
                    (loss / gradient_accumulation).backward()
                    # normalize the loss by gradient_accumulation step
                    if(global_step+1) % gradient_accumulation == 0:
                        torch.nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, model.generator.parameters()), max_grad_norm)
                        self.g_optimizer.step()
                        # anneal lr every update
                        #scaler.step(self.g_optimizer)
                        #scaler.update()
                        self.g_optimizer.zero_grad()
                
                else:
                    self.d_optimizer.zero_grad()
                    loss_d, y_hat, y_before = model.train_stage(mixture, source, stage, reset = reset)
                    (loss_d / gradient_accumulation).backward()
                    if(global_step+1) % gradient_accumulation == 0:
                        torch.nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, model.discriminator.parameters()), max_grad_norm)
                        self.d_optimizer.step()
                        #scaler.step(self.d_optimizer)
                        #scaler.update()
                        self.d_scheduler(self.d_optimizer)

                    
                    self.g_optimizer.zero_grad()
                    loss_g = model.train_stage(mixture, source, stage, y_hat = y_hat, y_before = y_before, reset = reset)
                    (loss_g / gradient_accumulation).backward()
                    if(global_step+1) % gradient_accumulation == 0:
                        torch.nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, model.generator.parameters()), max_grad_norm)
                        self.g_optimizer.step()
                        #scaler.step(self.g_optimizer)
                        #scaler.update()
                        self.g_scheduler(self.g_optimizer)
            
            else:
                with torch.no_grad():
                    if stage < 3:
                        loss = model.train_stage(mixture, source, stage, reset = reset)
                    else:
                        loss_d, y_hat, y_before = model.train_stage(mixture, source, stage, reset = reset)
                        loss_g = model.train_stage(mixture, source, stage, y_hat = y_hat, y_before = y_before, reset = reset)

            if stage == 3:
                loss_list = [loss_d.detach().item(), loss_g.detach().item()]
            else:
                loss_list = [loss.detach().item()]
            
            for i, name in enumerate(loss_names):
                history[name] += loss_list[i]

            if mode == 'train' and  (global_step+1) % avg_step == 0:
                if stage == 3:
                    print("{}, Epoch: {}, Step: {}\nloss_d: {}, loss_g: {}"
                        .format(mode, history['epoch'], history[step_name], history[loss_names[0]] / avg_step, history[loss_names[1]] / avg_step))
                else:
                    print("{}, Epoch: {}, Step: {}\nloss: {}"
                        .format(mode, history['epoch'], history[step_name], history[loss_names[0]] / avg_step))
                for name in loss_names:
                    self.writer.add_scalar(f'{mode}_loss/'+name, history[name] / avg_step, history[step_name])
                    history[name] = 0.0
            history[step_name] += 1
            global_step += 1
        
        if mode == 'dev':
            if stage < 3:
                this_loss = history['dev_loss'] / global_step 
            else:
                this_loss = history['dev_g'] / global_step
            if this_loss < history['dev_last_loss']:
                history['dev_last_loss'] = this_loss
                self.last_loss = this_loss
                module_names = ["Hifi_GAN"]
                optimizer_names = ["g_optimizer"]
                scheduler_names = ["g_scheduler"]
                if stage == 3:
                    optimizer_names += ["d_optimizer"]
                    scheduler_names += ["d_scheduler"]
                
                self.save_modules(stagename, module_names, optimizer_names, scheduler_names)
            for name in loss_names:
                self.writer.add_scalar(f'{mode}_loss/'+name, history[name] / global_step, history[step_name])
                history[name] = 0.0
            
            if stage < 3:
                self.g_scheduler.step(this_loss)
        return history


    def set_denoise(self, stage, resume, load_model):
        # module_names MUST have the same order with Hifi_GAN' parameters
        module_names = ["Hifi_GAN"]
        optimizer_names = ["g_optimizer"]
        scheduler_names = ["g_scheduler"]
        frozen_names = []
        devices = [1]
        stage_name = "stage" + str(self.stage)
        config = self.config[stage_name]

        self.set_device(module_names, frozen_names, devices)
        model = getattr(self, module_names[0])
        # #######################################
        # gpus = [str(x) for x in self.config['config']['gpu']]
        # os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(gpus)
        # dist.init_process_group(backend='nccl')
        # #torch.cuda.set_device(args.local_rank)
        # model = torch.nn.parallel.DistributedDataParallel(model)
        # ######################################
        scaler = GradScaler()

        lr = config['lr']
        n_warm_steps = config['n_warm_steps']
        self.g_optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.generator.parameters()), lr = lr, betas = (0.9,0.999))
        self.g_optimizer.zero_grad()
        self.g_scheduler = NoamScheduler(lr_initial = lr, n_warmup_steps = n_warm_steps)
        if stage == 3:
            lr_d = config['lr_d']
            self.d_optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.discriminator.parameters()), lr = lr_d, betas = (0.9,0.999))
            self.d_optimizer.zero_grad()
            self.d_scheduler = NoamScheduler(lr_initial = lr_d, n_warmup_steps = n_warm_steps)
            optimizer_names += ["d_optimizer"]
            scheduler_names += ["d_scheduler"]
        if resume or load_model:
            self.load_modules(stage_name, module_names, optimizer_names, scheduler_names, load_model=load_model)
        return model, scaler





if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='default')
    parser.add_argument('config_path', type=str, help='Config path of "*.yaml"')
    parser.add_argument('--gpu', type=int, nargs ='+', help='GPU available, such as "0 1"')
    parser.add_argument('--stage', default=1, type=int, help='Training Stage, [1,2,3]')
    parser.add_argument('--resume', default=False, type=bool, help='Saved chekpoints path to continue to train')
    parser.add_argument('--load_model', default=False, type=bool, help='Load model parameters for next stage')
    parser.add_argument('--user_defined_name', default='model', type=str, help='User defined name for save log and checkpoint')
    parser.add_argument('--local_rank', default=-1, type=int, help='node rank for distributed training')
    args = parser.parse_args()
    processor = Processor(args)
    processor.train(stage = args.stage, resume = args.resume, load_model= args.load_model)
