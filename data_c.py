# coding:utf8
import os
import torch
from torch.utils.data import DataLoader,Dataset
import numpy as np
import yaml
import librosa
import pandas as pd
import soundfile as sf
from utility import pad_sequence, collate_fn
from augment import AddNoise, AddReverb, SpeedPerturb, DropFreq, DropChunk, DoClip
from multichannel import MultiChannel
from utility import *

DEBUG = False
MAX_AMP = 0.95

class LibriPartyDataset(Dataset):
    def __init__(self):
        super(Dataset).__init__()
        global DEBUG
        with open('./config.yaml','r',encoding='utf-8') as f:
            config = f.read()
        config = yaml.load(config, Loader=yaml.FullLoader)
        self.config = config
        self.init_seed(config['config']['seed'])
        self.sample_rate = config['config']['sample_rate']
        
        self.spk_num = config['config']['spk_num']
        DEBUG = config['config']['DEBUG']

        #Read csv
        self.all_csv = self.get_csv(config['dataset'])
        self.buffer = []
        

    def set_attribute(self, dataset, augment = True, perturb = False, rir = False, noise = True, snr_low=0, snr_high=25):
        self.dataset = dataset
        self.batch_size = self.config['model']['batch_size']
        self.dataname = 'clean'
        self.csv = self.all_csv[self.dataset]
        self.size = len(self.csv[self.dataname])
        #Init augment
        self.do_augment = augment
        self.do_perturb = perturb
        self.do_rir = rir
        self.do_noise = noise
        if dataset == 'test':
            self.config['augment']['addnoise']['snr_low'] = snr_low
            self.config['augment']['addnoise']['snr_high'] = snr_high
        self.init_augment()


    def __len__(self):
        if self.dataset == 'train':
            return 30000 // self.batch_size * self.batch_size
        else:
            return 3000 // self.batch_size * self.batch_size
    
    def __getitem__(self, index):
        if len(self.buffer) > 0:
             mix, source, noise, length = self.get_buffer()
             flag = True
        else:
            while len(self.buffer) == 0:
                source, clean_length = [], []
                for _ in range(self.spk_num):
                    l = 0.0
                    while l < 16000:
                        rand_index = np.random.randint(self.size)
                        row = self.csv[self.dataname].iloc[rand_index]
                        path = row['path']
                        s, l = self.read_wav(path, self.sample_rate)
                    source += [s]
                    clean_length += [l]
                length = clean_length
                mix, source, noise, length = self.dynamic_mix(source, length)
                self.set_buffer(mix, source, noise, length)
            mix, source, noise, length = self.get_buffer()
            flag = False
        #SNR = self.compute_snr(mix[0], source[0,0])
        #print("SNR: ", SNR, " Length: ", length)
        data = {'mix':mix, 'source':source, 'noise':noise, 'length':length,'flag':flag}
        return data

    def init_seed(self, seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        torch.backends.cudnn.deterministic = True

    def init_augment(self):
        config = self.config['augment']
        self.addnoise = AddNoise(csv_file = self.csv['noise'], **config['addnoise'])
        self.addreverb = AddReverb(csv_file = self.csv['rir'], **config['addreverb'])
        self.perturb = SpeedPerturb(**config['perturb'])
        self.freqmask = DropFreq(**config['freqmask'])
        self.timemask = DropChunk(**config['timemask'])
        self.clip = DoClip(**config['clip'])
        self.single2multi = MultiChannel(**config['multichannel'])


    def get_csv(self, datas):
        '''
        Read csv files
        '''
        csv_set = {}
        datasets = ['train', 'dev', 'test']
        for dataset in datasets:
            data = datas[dataset]
            csv = {}
            for key in data.keys():
                csv[key] = self.read_csvset(data, key)
            csv_set[dataset] = csv
        return csv_set

    def read_csvset(self, data, key):
        '''
        Merge train csv set into a csv table. 
        '''
        csvset = []
        for path in data[key]:
            csv = pd.read_csv(path)
            csvset.append(csv)
        csv = pd.concat(csvset).drop_duplicates().dropna(axis=0, how='any')
        return csv


    def read_wav(self, path, sample_rate):
        sound, sr = sf.read(path)
        sound = librosa.resample(sound, sr, sample_rate)
        length = len(sound)
        return sound, length
    

    def compute_snr(self, estimation, origin, eps=1e-8):
        estimation = estimation - torch.mean(estimation, -1, keepdim=True)
        origin = origin - torch.mean(origin, -1, keepdim=True)
   
        def calculate(estimation, origin):
            origin_power = torch.pow(origin, 2).sum(-1, keepdim=True) + eps  # (batch, 1)
            scale = torch.sum(origin*estimation, -1, keepdim=True) / origin_power  # (batch, 1)

            est_true = scale * origin  # (batch, nsample)
            est_res = estimation - est_true  # (batch, nsample)

            true_power = torch.pow(est_true, 2).sum(-1) + eps
            res_power = torch.pow(est_res, 2).sum(-1) + eps

            return 10*torch.log10(true_power) - 10*torch.log10(res_power)  # (batch, )
            
        snr = torch.mean(calculate(estimation, origin))
        return snr

    def set_buffer(self, mix, source, noise, length):
        num_spk = self.config['config']['spk_num']
        max_length = self.config['config']['max_length']
        if self.dataset == 'test':
            self.buffer.append([mix, source, noise, length])
            return
        
        start = 0
        lens = mix.shape[-1]
        while start < lens:
            l = np.random.randint(16000, max_length)
            #l = max_length
            end = min(lens, start+l)
            length[0] = end-start
            #SNR = self.compute_snr(mix[0,start:end], source[0,0,start:end])
            if length[0] < 16000:
                break
            self.buffer.append([mix[...,start:end], source[...,start:end], noise[...,start:end],length.clone()])
            start += end


    def get_buffer(self):
        mix, source, noise, length = self.buffer.pop()
        return mix.float(), source.float(), noise.float(), length


    def removeSilence(self, x, dyn_range=40, N=256, K=128):
        w = torch.unsqueeze(torch.from_numpy(np.hanning(256)), 0).to(torch.float)
        X1 = x[0 : int(x.shape[0]) // N * N].reshape(int(x.shape[0]) // N, N).T
        X2 = (
            x[128 : (int(x.shape[0]) - 128) // N * N + 128]
            .reshape((int(x.shape[0]) - 128) // N, N)
            .T
        )
        X = torch.zeros(N, X1.shape[1] + X2.shape[1])
        X[:, 0::2] = X1
        X[:, 1::2] = X2

        energy = 20 * torch.log10(
            torch.sqrt(torch.matmul(w ** 2, X ** 2)) / 16.0 + smallVal
        )

        Max_energy = torch.max(energy)
        msk = torch.squeeze((energy - Max_energy + dyn_range > 0))
        x_sil = w.T.repeat(1, X[:, msk].shape[-1]) * X[:, msk]
        x_sil = torch.cat(
            (
                x_sil[0:128, 0],
                (x_sil[0:128, 1:] + x_sil[128:, 0:-1]).T.flatten(),
                x_sil[128:256, -1],
            ),
            axis=0,
        )
        return x_sil

    def dynamic_mix(self, source_list, length):
        eps = 1e-10
        augment_source = []
        for i, source in enumerate(source_list):
            source = torch.tensor(source, dtype=torch.float32)
            #source = self.removeSilence(source, dyn_range=20, N=256, K=128)
            if self.do_perturb:
                source = self.perturb(source)
            source_list[i] = source
            # if self.do_rir:
            #     source = self.addreverb(source)
            if self.do_augment:
                source = self.freqmask(source)
                source = self.timemask(source)
                source = self.clip(source)
            augment_source += [source]
        
        #Tranform single channel to multichannel, shape: src, chan, T
        #param = self.single2multi.get_param()
        #source, augment_source = self.single2multi.simulate(source_list, augment_source, param)

        # augment_source = self.addreverb(augment_source[0])
        # augment_source = torch.stack([augment_source]*3, dim=0)
        # augment_source = [augment_source]

        source, augment_source, noise_rir = self.single2multi.simulate(source_list, augment_source, noise=True)
        for i, s in enumerate(augment_source):
            length[i] = s.shape[-1]

        augment_source = pad_sequence(augment_source, pad_value=0)
        mix = torch.sum(augment_source, dim=0)
        #Add Noise
        if self.do_noise:
            mix, noise = self.addnoise(mix.transpose(1,0), self.single2multi, noise_rir)
            #mix, noise = self.addnoise(mix.transpose(1,0))
            #mix, noise = self.addnoise(mix.transpose(1,0), self.single2multi, param)
            mix = mix.transpose(1,0)
            noise = noise.transpose(1,0)

        if torch.abs(mix).max() >= MAX_AMP:
            mix = mix * MAX_AMP / (torch.abs(mix).max() + eps)
        length = torch.tensor(length, dtype=torch.long)
        return mix, augment_source, noise, length
    
    

def generate_testdataset():
    dataset = LibriPartyDataset()
    dataset.set_attribute('test',augment=False)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0, collate_fn=collate_fn)
    for i,data in enumerate(dataloader):
        SNR = dataset.compute_snr(data['mix'][0,0], data['source'][0,0,0])
        print(f'data batch {i}: {SNR} dB')
        sf.write(f'Chinese_data/noisy/{i}_{SNR}.wav', data['mix'][0,0].numpy(), 16000)
        sf.write(f'Chinese_data/ref/{i}_{SNR}.wav', data['source'][0,0,0].numpy(), 16000)
        # for key in data.keys():
        #     print("{}: {}".format(key, data[key].shape))
        

if __name__ == '__main__':
    # generate_testdataset()
    dataset = LibriPartyDataset()
    dataset.set_attribute('train',augment=False)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0, collate_fn=collate_fn)

    # from tqdm import tqdm
    # all_csv = dataset.all_csv['train']['noise']
    # print(all_csv.shape[0])
    # lens = 0.0
    # for index in tqdm(range(all_csv.shape[0])):
    #     row = all_csv.iloc[index]
    #     path = row['path']
    #     s, l = dataset.read_wav(path, 16000)
    #     lens += l
    # print(lens / 16000)


    snr = 0.0
    stoi = 0.0
    lres = []
    lreal = []
    for i,data in enumerate(dataloader):
        print('data batch {}'.format(i))
        s = torch.stft(data['source'][0,0,0], n_fft=400, hop_length=160, return_complex=True).abs()
        
        energy = 20 * torch.log10(torch.mean(s, dim=0) + 1e-8) - 10 * torch.log10(torch.tensor(1e-8))
        min_energy = torch.min(energy)
        mean_energy = torch.mean(energy)
        print(mean_energy, min_energy)
        # if mean_energy < 0:
        #     sf.write(f'sample/{i}_{mean_energy}.wav', data['source'][0,0,0].numpy(), 16000)

        SNR =  dataset.compute_snr(data['mix'][0,0], data['source'][0,0,0])
        print(data['length'], data['mix'][:,0].shape)
        snr += SNR
        STOI = -stoi_loss(data['mix'][:,0], data['source'][:,0,0], data['length'])
        stoi += STOI
        print("SNR: ", SNR, "STOI: ", STOI)
        # for key in data.keys():
        #     print("{}: {}".format(key, data[key].shape))

        # import matplotlib.pyplot as plt
        # import librosa.feature
        # import librosa.display
        # import librosa
        # def plot_melspectrogram(wav, sr, n_mels=80, fmax=8000):
        #     fig, ax = plt.subplots(1,1,figsize=(20,5))
        #     M = librosa.feature.melspectrogram(y=wav, sr=sr, n_mels=n_mels, fmax=fmax, n_fft=400)
        #     M_db = librosa.power_to_db(M, ref=np.max)
        #     img = librosa.display.specshow(M_db, sr=sr, y_axis='mel', x_axis='time', ax=ax, fmax=fmax)
        #     ax.set(title='Mel spectrogram display')
        #     fig.colorbar(img, ax=ax, format="%+2.f dB")
        #     plt.savefig('./tmp.png')
        # # plot_melspectrogram(data['source'][0,0,0].numpy(), 16000)
        # # sf.write(f'tmp.wav', data['source'][0,0,0].numpy(), 16000)

        # from pesq import pesq
        # from pystoi.stoi import stoi
        # from metrics import SI_SDR
        # res = pesq_loss(data['source'][:,0,0], data['mix'][:,0], data['length'])
        # #res = stoi(data['source'][0,0,0].numpy(), data['mix'][0,0].numpy(), 16000, extended=False)
        # #res = SI_SDR(data['source'][0,0,0].numpy(), data['mix'][0,0].numpy())
        # real_res = pesq(16000, data['source'][0,0,0].numpy(), data['mix'][0,0].numpy(), 'wb')
        # print(f"real pesq : {real_res}, pesq : {res}")
        # lres.append(res)
        # lreal.append(real_res)
        # if i>=100:
        #     lres = np.array(lres)
        #     lreal = np.array(lreal)
        #     lres -= np.mean(lres)
        #     lreal -= np.mean(lreal)
        #     coef = np.mean(lres * lreal) / np.sqrt(np.mean(lres**2)) / np.sqrt(np.mean(lreal**2))
        #     print(coef)
        #     break

        
        
        
