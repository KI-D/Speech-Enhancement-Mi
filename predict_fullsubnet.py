import os
import torch
import argparse
from fullsubnet import FullSubNet
from data_c import LibriPartyDataset
from torch.utils.data import DataLoader
import yaml
import numpy as np
import soundfile as sf
#import speechmetrics
from utility import *
from metrics import *

def predict(args):
    with open(args.config_path,'r',encoding='utf-8') as f:
        config = f.read()
    config = yaml.load(config, Loader=yaml.FullLoader)
    config['config']['gpu'] = args.gpu
    config['user_defined_name'] = args.user_defined_name
    sample_rate = config['config']['sample_rate']
    # gpus = [str(x) for x in config['config']['gpu']]
    # os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(gpus)

    stage_dir = os.path.join('modules/denoise',config['user_defined_name'])
    name = args.name
    batch_size = config['model']['batch_size']
    num_workers = config['denoise']['num_workers']
    max_length = config['config']['max_length']
    num_mic = config['config']['num_mic']

    model = globals()[name](**config[name]).cuda()
    path = os.path.join(stage_dir, name+'.pth')
    model.load_state_dict(torch.load(path))

    data = LibriPartyDataset(snr_low=0, snr_high=25)
    data.set_attribute('test', augment = False, perturb=False)
    dataloader = DataLoader(data, batch_size = 1, shuffle=True, num_workers=num_workers)
    index = 0
    score_delta = {}
    score_before = {}
    score_after = {}
    for data in dataloader:
        mixture = data['mix'].cuda()
        source = data['source'].cuda().squeeze(1)
        noise = data['noise'].cuda()
        length = data['length'].cuda()

        # if index < 187:
        #     index += 1
        #     continue
        # if index > 187:
        #     index += 1
        #     break
        
        with torch.no_grad():
            model.eval()
            # start = 0
            # lens = mixture.shape[-1]
            # N = (lens - 1) // 64000 + 1
            # left = N * 64000 - lens
            # if left != 0:
            #     pad = torch.zeros((1, 3, left), dtype=mixture.dtype, device=mixture.device)
            #     mix = torch.cat([mixture,pad], dim = -1)
            #     src = torch.cat([source, pad], dim = -1)
            # separated = []
            # flag = False
            # for i in range(N):
            #     sep, _, _, _ = model.realtime_process(mix[...,start:start+64000], src[...,start:start+64000], flag=flag, train=False)
            #     separated.append(sep)
            #     start += 64000
            #     flag = True
            # separated = torch.cat(separated, dim=-1)
            # if left:
            #     separated = separated[...,:lens]
            separated, _, _, _ = model.realtime_process(mixture, source, flag=False, train=False)

        separated = separated.detach().cpu().numpy().squeeze()
        mixture = mixture[:,0].detach().cpu().numpy().squeeze()
        source = source[:,0].detach().cpu().numpy().squeeze()
        #window_length = length.cpu().numpy()/sample_rate
        #metrics = speechmetrics.load(['mosnet','srmr','bsseval','pesq','stoi','sisdr'], window_length)
        #score_tgt = metrics(separated, source.squeeze(), rate=sample_rate)
        #score_src = metrics(mixture, source.squeeze(), rate=sample_rate)
        
        score_tgt = {}
        score_src = {}

        #score_tgt['SDR'] = SDR(source, separated)
        score_tgt['SI_SDR'] = SI_SDR(source, separated)
        score_tgt['STOI'] = STOI(source, separated)
        score_tgt['WB_PESQ'] = WB_PESQ(source, separated)
        score_tgt['NB_PESQ'] = NB_PESQ(source, separated)

        #score_src['SDR'] = SDR(source, mixture)
        score_src['SI_SDR'] = SI_SDR(source, mixture)
        score_src['STOI'] = STOI(source, mixture)
        score_src['WB_PESQ'] = WB_PESQ(source, mixture)
        score_src['NB_PESQ'] = NB_PESQ(source, mixture)
        for key in score_tgt.keys():
            if key in score_delta:
                score_delta[key] += score_tgt[key]-score_src[key]
                score_before[key] += score_src[key]
                score_after[key] += score_tgt[key]
            else:
                score_delta[key] = score_tgt[key]-score_src[key]
                score_before[key] = score_src[key]
                score_after[key] = score_tgt[key]

        print(f"Number: {index},\n Source: {score_src},\n Target: {score_tgt}")

        # if score_tgt['STOI']-score_src['STOI'] <= -0.05:
        #     amp = np.iinfo(np.int16).max
        #     mixture = np.int16(0.8 * amp * mixture / np.max(np.abs(mixture)))
        #     separated = np.int16(0.8 * amp * separated / np.max(np.abs(separated)))
        #     save_wave('Mixed'+str(index)+'_', mixture)
        #     save_wave('Separated'+str(index)+'_', separated)
        
        index += 1

    for key in score_delta.keys():
        print(f"{key}: {score_delta[key]/index}")
        print(f"{key}: {score_before[key]/index}")
        print(f"{key}: {score_after[key]/index}")
        print('\n')
            



def save_wave(name, data):
    rate = 16000
    if not os.path.exists('result'):
        os.makedirs('result')
    if len(data.shape) == 1:
        data = data[None, :]
    for i in range(len(data)):
        sf.write('result/' + name + str(i) + '.wav', data[i], rate)




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='default')
    parser.add_argument('config_path', type=str, help='Config path of "*.yaml"')
    parser.add_argument('name', type=str, help='Algorithm Name')
    parser.add_argument('--gpu', type=int, nargs ='+', help='GPU available, such as "0 1"')
    parser.add_argument('--stage', default=0, type=int, help='Training Stage, 0 for denoise')
    parser.add_argument('--user_defined_name', default='model', type=str, help='User defined name for save log and checkpoint')
    args = parser.parse_args()

    predict(args)