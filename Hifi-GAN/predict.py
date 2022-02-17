import os
import sys
sys.path.append("../")
import torch
import argparse
from hifigan import Hifi_GAN
from data_c import LibriPartyDataset
from torch.utils.data import DataLoader
import yaml
import numpy as np
import soundfile as sf
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
    stage_name = "stage" + str(args.stage)
    stage_dir = os.path.join('modules',stage_name,config['user_defined_name'])
    name = args.name
    batch_size = config['model']['batch_size']
    num_workers = config[stage_name]['num_workers']
    max_length = config['config']['max_length']
    num_mic = config['config']['num_mic']

    model = globals()[name](**config[name]).cuda()
    path = os.path.join(stage_dir, name+'.pth')
    model.load_state_dict(torch.load(path), strict=False)

    data = LibriPartyDataset()
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


        with torch.no_grad():
            model.eval()
            separated = model(mixture)

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
        #score_tgt['NB_PESQ'] = NB_PESQ(source, separated)

        #score_src['SDR'] = SDR(source, mixture)
        score_src['SI_SDR'] = SI_SDR(source, mixture)
        score_src['STOI'] = STOI(source, mixture)
        score_src['WB_PESQ'] = WB_PESQ(source, mixture)
        #score_src['NB_PESQ'] = NB_PESQ(source, mixture)
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
    parser.add_argument('--stage', default=0, type=int, help='Training Stage, 0 for default')
    parser.add_argument('--user_defined_name', default='model', type=str, help='User defined name for save log and checkpoint')
    args = parser.parse_args()

    predict(args)