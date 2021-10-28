import gpuRIR
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
import torch
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '4'
class MultiChannel(object):
    def __init__(self, room_limit, t60_limit, beta_limit, array_limit, mic_limit, source_limit, num_src, num_mic, fs):
        """
        room_limit: 房间的大小范围，形如 ((xl,yl,zl),(xh,yh,zh))
        t60_limit: 停止发声后衰减60DB的时间范围， 形如 (tl, th)
        beta_limit: 房间墙壁反射系数范围， 形如 ((x1,x2,x3,x4,x5,x6),(y1,y2,y3,y4,y5,y6))
        array_limit: 麦克风阵列中心在房间内的位置范围(0-1)
        mic_limit: 麦克风距离阵列中心的位置(绝对值以米为单位)
        source_limit: 声源位置范围(0-1)
        fs: 采样率
        """
        self.room_limit = room_limit
        self.t60_limit = t60_limit
        self.beta_limit = beta_limit
        self.array_limit = array_limit
        self.mic_limit = mic_limit
        self.source_limit = source_limit
        self.num_src = num_src
        self.num_mic = num_mic
        self.fs = fs
    
    def sample_ND(self, low, high, size = 3):
        assert len(low) == size and len(high) == size
        low = np.array(low)
        high = np.array(high)
        rpos = np.random.rand(size) * (high-low) + low
        return rpos
    
    def simulate(self, sources, aug_sources=None, noise=False, RIR=None):
        if RIR is None:
            room_xyz =  self.sample_ND(*self.room_limit)
            t60 = np.random.rand() * (self.t60_limit[1] - self.t60_limit[0]) + self.t60_limit[0]
            beta =  self.sample_ND(*self.beta_limit, 6)
            #nb_img: Image方法阶数
            if t60 == 0:
                Tdiff = 0.1
                Tmax = 0.1
                nb_img = [1, 1, 1]
            else:
                Tdiff = gpuRIR.att2t_SabineEstimator(15, t60)
                Tmax = gpuRIR.att2t_SabineEstimator(60, t60)
                if t60 < 0.15: 
                    Tdiff = Tmax
                nb_img = gpuRIR.t2n(Tdiff, room_xyz)

            mic_pos = np.zeros((self.num_mic, 3))
            array_pos = self.sample_ND(*self.array_limit) * room_xyz
            for i in range(self.num_mic):
                mic_pos[i, :] = array_pos +  self.sample_ND(*self.mic_limit)
            
            multichannel = []
            aug_multichannel = []
            if type(sources[0]) == torch.Tensor:
                sources = [x.numpy() for x in sources]
                if aug_sources is not None:
                    aug_sources = [x.numpy() for x in aug_sources]
            else:
                sources = [np.array(x) for x in sources]
                if aug_sources is not None:
                    aug_sources = [np.array(x) for x in aug_sources]
            if noise:
                num_src = self.num_src+1
            else:
                num_src = self.num_src

            for i in range(num_src):
                # 声源的坐标
                source_pos = self.sample_ND(*self.source_limit) * room_xyz
                source_pos = source_pos.reshape(1,-1)
                # 生成RIR
                RIR = gpuRIR.simulateRIR(
                    room_sz=room_xyz,
                    beta=beta,
                    nb_img=nb_img,
                    fs=self.fs,
                    pos_src=source_pos,
                    pos_rcv=mic_pos,
                    Tmax=Tmax,
                    Tdiff=Tdiff,
                    mic_pattern='omni'
                )
                if i >= self.num_src:
                    break
                # 生成多通道语音
                multichannel += [torch.tensor(np.transpose(gpuRIR.simulateTrajectory(sources[i], RIR, fs=self.fs), (1,0)))]
                if aug_sources is not None:
                    aug_multichannel += [torch.tensor(np.transpose(gpuRIR.simulateTrajectory(aug_sources[i], RIR, fs=self.fs), (1,0)))]
            # sf.write('multi.wav', multichannel[0].transpose(1,0).numpy(), self.fs)
            if noise:
                return multichannel, aug_multichannel, RIR
            else:
                return multichannel, aug_multichannel
        else:
            noise_multi = torch.tensor(np.transpose(gpuRIR.simulateTrajectory(sources, RIR, fs=self.fs), (1,0)))
            return noise_multi

if __name__ == "__main__":
    room = ((3,3,2.5), (4,5,3))
    t60 = (0.2, 1.0)
    beta = ((0.5,0.5,0.5,0.5,0.5,0.5), (1.0,1.0,1.0,1.0,1.0,1.0))
    array = ((0.1, 0.1, 0.2), (0.9, 0.9, 0.7))
    mic = ((0.06, 0.06, 0.06), (0.15, 0.15, 0.15))
    source = ((0.0, 0.0, 0.3), (1.0, 1.0, 0.7))
    test_code = MultiChannel(room_limit=room, t60_limit=t60, beta_limit=beta, array_limit=array, mic_limit=mic, source_limit=source, num_src = 2, num_mic = 3, fs=16000)
    source, sr = sf.read('sample.flac')
    source = [source,source]
    test_code.simulate(source, source)
