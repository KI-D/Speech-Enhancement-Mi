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
from fullsubnet import FullSubNet

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
        self.stage2str = ['denoise']

        self.dataset = LibriPartyDataset()
        self.init_seed(config['config']['seed'])
        self.init_modules()


    def init_seed(self, seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        torch.backends.cudnn.deterministic = True

    def init_modules(self):
        module_names = ["FullSubNet"]
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
    
    def save_modules(self, stage_name, module_names):
        check_dir = self.config['config']['checkpoint_dir']
        stage_dir = os.path.join(check_dir, stage_name, self.config['user_defined_name'])
        if not os.path.exists(stage_dir):
            os.makedirs(stage_dir)
        
        for name in module_names:
            path = os.path.join(stage_dir, name+'.pth')
            torch.save(getattr(self, name).state_dict(), path)
    
    def load_modules(self, stage_name, module_names, resume):
        if stage_name == "fronzen":
            stage_dir = "fronzen_modules"
        else:
            stage_dir = os.path.join(self.config['config']['checkpoint_dir'], stage_name, resume)
        
        for name in module_names:
            path = os.path.join(stage_dir, name+'.pth')
            getattr(self, name).load_state_dict(torch.load(path), strict=False)


    def train(self, stage, resume = None):
        stage_name = self.stage2str[stage]
        num_epoch = self.config[stage_name]['num_epoch']
        if stage == 0:
            history = {"train_step":0, "dev_step":0, "epoch":0,  "train_loss":0.0, "train_logmse":0.0, "train_sisnr":0.0, 
                    "dev_last_loss":1e8, "dev_loss":0.0, "dev_logmse":0.0, "dev_sisnr":0.0}
            model, optimizer, scheduler, scaler = self.set_denoise(stage, resume)
            self.set_writter(stage_name)

            for epoch in range(num_epoch):
                history["epoch"] = epoch
                history = self.denoise(model, optimizer, scheduler, scaler, stage, "train", history)
                history = self.denoise(model, optimizer, scheduler, scaler, stage, "dev", history)
        


    def denoise(self, model, optimizer, scheduler, scaler, stage, mode, history):
        stage_name = self.stage2str[stage]
        config = self.config[stage_name]
        max_grad_norm = self.config['config']['max_grad_norm']
        avg_step = self.config['config']['avg_step']
        batch_size = config['batch_size']
        num_workers = config['num_workers']
        num_mic = config['num_mic']
        gradient_accumulation = config['gradient_accumulation']

        self.dataset.set_attribute(mode, augment = False)
        self.dataset.buffer = []
        # ################################################################################
        # sampler = torch.utils.data.distributed.DistributedSampler(self.dataset)
        # dataloader = DataLoader(self.dataset, batch_size = batch_size, shuffle=True, collate_fn=collate_fn, num_workers=num_workers, pin_memory = True, sampler=sampler)
        # ################################################################################
        dataloader = DataLoader(self.dataset, batch_size = batch_size, shuffle=True, num_workers=num_workers)
        step_name = mode + '_step'
        loss_names = [mode + '_loss', mode + '_logmse', mode + '_sisnr']
        global_step = 0.0
        for data in dataloader:
            device = 'cuda:1'
            mixture = data['mix'].to(device)
            source = data['source'].to(device).squeeze(1)
            noise = data['noise'].to(device)
            length = data['length'].to(device)
            num_mics = torch.tensor([num_mic]*batch_size).long().to(device)
            ##########################################################################
            # mixture = data['mix'].cuda()
            # source = data['source'].cuda()
            # noise = data['noise'].cuda()
            # length = data['length'].cuda()
            # spk = data['spk'].cuda()
            # num_mics = torch.tensor([num_mic]*batch_size).long().cuda()
            ##########################################################################
            if mode == 'train':
                pred_source, pred_crm, sf, xf = model.realtime_process(mixture, source, data['flag'], train=False)
                loss, logmse, sisnr = model.compute_loss(source[:,0], pred_source, xf, sf, pred_crm, length)
                # normalize the loss by gradient_accumulation step
                scaler.scale(loss / gradient_accumulation).backward()
                if(global_step+1) % gradient_accumulation == 0:
                    torch.nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, model.parameters()), max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                    # anneal lr every update
                    # scheduler(optimizer)
            else:
                with torch.no_grad():
                    pred_source, pred_crm, sf, xf = model.realtime_process(mixture, source, data['flag'], train=False)
                    loss, logmse, sisnr = model.compute_loss(source[:,0], pred_source, xf, sf, pred_crm, length)
            
            loss_list = [loss.detach().item(), logmse.detach().item(), sisnr.detach().item()]
            for i, name in enumerate(loss_names):
                history[name] += loss_list[i]

            if mode == 'train' and  (global_step+1) % avg_step == 0:
                print("{}, Epoch: {}, Step: {}\nloss: {}, logmse:{}, sisnr:{}"
                    .format(mode, history['epoch'], history[step_name], history[loss_names[0]] / avg_step, history[loss_names[1]] / avg_step, history[loss_names[2]] / avg_step))
                for name in loss_names:
                    self.writer.add_scalar(f'{mode}_loss/'+name, history[name] / avg_step, history[step_name])
                    history[name] = 0.0
            history[step_name] += 1
            global_step += 1
        
        if mode == 'dev':
            this_loss = history['dev_loss'] / global_step
            if this_loss < history['dev_last_loss']:
                module_names = ["FullSubNet"]
                self.save_modules(stage_name, module_names)
            for name in loss_names:
                self.writer.add_scalar(f'{mode}_loss/'+name, history[name] / global_step, history[step_name])
                history[name] = 0.0
            history['dev_last_loss'] = this_loss
            scheduler.step(this_loss)
        return history


    def set_denoise(self, stage, resume = None):
        # module_names MUST have the same order with FullSubNet' parameters
        module_names = ["FullSubNet"]
        frozen_names = []
        devices = [1]
        stage_name = self.stage2str[stage]
        config = self.config[stage_name]
        lr = config['lr']
        n_warm_steps = config['n_warm_steps']

        self.set_device(module_names, frozen_names, devices)
        model = getattr(self, module_names[0])
        # #######################################
        # gpus = [str(x) for x in self.config['config']['gpu']]
        # os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(gpus)
        # dist.init_process_group(backend='nccl')
        # #torch.cuda.set_device(args.local_rank)
        # model = torch.nn.parallel.DistributedDataParallel(model)
        # ######################################
        if resume is not None:
            self.load_modules(stage_name, module_names, resume)
        scaler = GradScaler()
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr = lr, betas = (0.9,0.999))
        optimizer.zero_grad()
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True, min_lr=1e-7)
        #scheduler = NoamScheduler(lr_initial = lr, n_warmup_steps = n_warm_steps, model_size = config['model_dim'])
        return model, optimizer, scheduler, scaler




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='default')
    parser.add_argument('config_path', type=str, help='Config path of "*.yaml"')
    parser.add_argument('--gpu', type=int, nargs ='+', help='GPU available, such as "0 1"')
    parser.add_argument('--stage', default=0, type=int, help='Training Stage, 0 for denoise')
    parser.add_argument('--resume', default=None, type=str, help='Saved chekpoints path to continue to train')
    parser.add_argument('--user_defined_name', default='model', type=str, help='User defined name for save log and checkpoint')
    parser.add_argument('--local_rank', default=-1, type=int, help='node rank for distributed training')
    args = parser.parse_args()
    processor = Processor(args)
    processor.train(stage = args.stage, resume = args.resume)
