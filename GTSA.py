import torch
from torch.nn import functional
from torch import nn
from torch.cuda.amp import autocast as autocast
import torch.nn.init as init
from torch_complex import ComplexTensor
import torch_complex
from utility import *
from speechbrain.processing.features import STFT,ISTFT



class GlobalLayerNorm(nn.Module):
    '''
    import torch
    from modules import GlobalLayerNorm
    norm = GlobalLayerNorm(201)
    x = torch.rand(3, 201, 6)
    y = norm(x)
    print(x.shape, norm.step)
    '''
    def __init__(self, dim, last = False, time = True):
        super(GlobalLayerNorm, self).__init__()
        self.time = time
        
        if time:
            self.register_buffer('step', torch.tensor(0))
            self.register_buffer('mean', torch.tensor(0))
            self.register_buffer('var', torch.tensor(0))
        
        if last:
            self.weight = nn.Parameter(torch.ones(1, 1, dim))
            self.bias = nn.Parameter(torch.zeros(1, 1, dim))
        else:
            self.weight = nn.Parameter(torch.ones(1, dim, 1))
            self.bias = nn.Parameter(torch.zeros(1, dim, 1))


    def forward(self, x):
        # x = B x F x T
        T = x.shape[-1]
        if x.dim() == 4:
            weight = self.weight.unsqueeze(-1)
            bias = self.bias.unsqueeze(-1)
            mean = torch.mean(x, (1,2,3), keepdim=True)
            var = torch.mean((x-mean)**2, (1,2,3), keepdim=True)
        else:
            weight = self.weight
            bias = self.bias
            mean = torch.mean(x, (1,2), keepdim=True)
            var = torch.mean((x-mean)**2, (1,2), keepdim=True)


        if self.time:
            alpha = self.step / (self.step + T)
            global_mean = alpha * self.mean + (1.0 - alpha) * mean
            global_var = alpha * (self.var + (global_mean - self.mean) ** 2) + (1.0 - alpha) * (var + (global_mean - mean) ** 2)
        else:
            global_mean = mean
            global_var = var

        x = (x - global_mean) / (torch.sqrt(global_var) + EPS) * weight + bias
        if self.time:
            with torch.no_grad():
                self.step += T
                self.mean = global_mean.clone()
                self.var = global_var.clone()
        return x
    
    def reset(self):
        if self.time:
            self.mean = torch.zeros_like(self.mean)
            self.var = torch.zeros_like(self.var)
            self.step = torch.zeros_like(self.step)



class MutiheadAttention(nn.Module):
    '''
    import torch
    from modules import MutiheadAttention
    attention = MutiheadAttention(3, 201, 500, batch_size=3)
    x = torch.rand(3, 6, 201)
    y = attention(x)
    print(x.shape, attention.bk[0,:,0], attention.bv[0,:,0])
    '''
    def __init__(self, num_heads, model_dim, maxlen, batch_size = 1):
        super(MutiheadAttention, self).__init__()
        self.num_heads = num_heads
        self.model_dim = model_dim
        self.maxlen = maxlen
        self.batch_size = batch_size
        
        self.ql = nn.Linear(model_dim, model_dim)
        self.kl = nn.Linear(model_dim, model_dim)
        self.vl = nn.Linear(model_dim, model_dim)
        self.linear = nn.Linear(model_dim, model_dim)
        self.softmax = nn.Softmax(dim=-1)
        
        self.delta = nn.Parameter(torch.ones(1))
        self.d = torch.sqrt(torch.tensor(self.model_dim).type(torch.float32))


        self.register_buffer('bk', torch.zeros((batch_size * num_heads, maxlen, model_dim // num_heads)))
        self.register_buffer('bv', torch.zeros((batch_size * num_heads, maxlen, model_dim // num_heads)))


    def forward(self, x):
        # x = B x T x F
        B, T, F = x.shape
        q = torch.cat(torch.split(self.ql(x), F // self.num_heads, dim=-1), dim=0) 
        k = torch.cat(torch.split(self.kl(x), F // self.num_heads, dim=-1), dim=0)
        v = torch.cat(torch.split(self.vl(x), F // self.num_heads, dim=-1), dim=0)
        
        k = torch.cat([self.bk[:, T:], k], dim=1)
        v = torch.cat([self.bv[:, T:], v], dim=1)
        ind = torch.arange(1, self.maxlen+1).unsqueeze(1).repeat(1,self.maxlen).to(x.device)
        G = torch.exp(-(ind-ind.transpose(0,1))**2 / (self.delta**2 + EPS)).unsqueeze(0)
        #[E*B, T, F/E] * [E*B, F/E, maxlen]
        x = torch.matmul(q, k.transpose(1,2)) * G[:, -T:] / self.d
        #[E*B, T, maxlen] * [E*B, maxlen, F/E]
        x = torch.matmul(self.softmax(x), v)

        x = torch.cat(torch.split(x, B, dim=0), dim=-1)
        x = self.linear(x)

        with torch.no_grad():
            self.bk = k.clone()
            self.bv = v.clone()
        return x
    
    def reset(self):
        self.bk = torch.zeros_like(self.bk)
        self.bv = torch.zeros_like(self.bv)


class TransformerLayer(nn.Module):
    '''
    import torch
    from modules import TransformerLayer
    transformer = TransformerLayer(3, 201, 500)
    x = torch.rand(3, 201, 6)
    y = transformer(x)
    print(x.shape)
    '''
    def __init__(self, num_heads, model_dim, fn_dim, maxlen=500, dropout=0.0):
        super(TransformerLayer, self).__init__()
        self.attention = MutiheadAttention(num_heads, model_dim, maxlen)
        self.norm_a =  GlobalLayerNorm(model_dim, last=True, time=False)
        self.activation = nn.ReLU()
        self.linear_in = nn.Linear(model_dim, fn_dim)
        self.linear_out = nn.Linear(fn_dim, model_dim)
        self.norm_i = GlobalLayerNorm(model_dim, last=True, time=False)
        self.dropout = nn.Dropout(dropout)


    def forward(self, x):
        # x = B x F x T
        #[B, T, F]
        x = x.transpose(1,2)
        x = self.attention(x) + x
        x = self.norm_a(x)
        res = x
        x = self.activation(self.linear_in(x))
        x = self.dropout(x)
        x = self.linear_out(x) + res
        x = self.norm_i(x).transpose(1,2)
        return x
    
    def reset(self):
        self.attention.reset()
        self.norm_a.reset()
        self.norm_i.reset()



class TemporalConv2d(nn.Module):
    """
    ---------------------------------------
    import torch
    from modules import TemporalConv2d
    model = TemporalConv2d(1, 16, 3, 1, (1,2), (1,4))
    x = torch.rand(3, 1, 20, 10)
    y = model(x)
    print(x.shape, model.buffer[0, 0, 0, :])
    """
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.0, activation="ReLU"):
        super(TemporalConv2d, self).__init__()
        self.padding = padding[1]

        if activation == "Tanh":
            self.activation = nn.Tanh()
        elif activation == "ReLU":
            self.activation = nn.ReLU()
        elif activation == "Sigmoid":
            self.activation = nn.Sigmoid()
        else:
            raise NotImplementedError(f"Not implemented activation function {activation}")
        
        #手动padding
        self.conv = nn.Conv2d(n_inputs, n_outputs, kernel_size, stride=stride, padding=(padding[0], 0), dilation=dilation)
        self.dropout = nn.Dropout(dropout)
        self.net = nn.Sequential(self.conv, self.dropout)
        self.norm = GlobalLayerNorm(n_outputs, time=False)
        self.register_buffer('buffer', None)

    def forward(self, x):
        #x: [B, C, F, T]
        #return: [B, C', F, T]
        B, C, F, T = x.shape
        if self.buffer is None:
            with torch.no_grad():
                self.buffer = torch.zeros((B, C, F, self.padding), device=x.device, dtype=x.dtype)

        inp = torch.cat([self.buffer, x], dim=-1)
        out = self.net(inp)
        out = self.activation(out) 
        out = self.norm(out)
        with torch.no_grad():
            if T > self.padding:
                self.buffer = x[..., -self.padding:]
            else:
                self.buffer[..., :self.padding - T] = self.buffer[..., T-self.padding:]
                self.buffer[..., -T:] = x
        return out
    
    def reset(self):
        self.buffer = None
        self.norm.reset()


class TemporalConvTranspose2d(nn.Module):
    """
    ---------------------------------------
    import torch
    from modules import TemporalConvTranspose2d
    model = TemporalConvTranspose2d(16, 1, 3, 1, (1,2), (1,4))
    x = torch.rand(3, 16, 20, 10)
    res = torch.rand(3, 1, 20, 10)
    y = model(x, res)
    """
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.0, activation="ReLU"):
        super(TemporalConvTranspose2d, self).__init__()
        self.padding = padding[1]

        if activation == "Tanh":
            self.activation = nn.Tanh()
        elif activation == "ReLU":
            self.activation = nn.ReLU()
        elif activation == "Sigmoid":
            self.activation = nn.Sigmoid()
        else:
            raise NotImplementedError(f"Not implemented activation function {activation}")
        
        #手动padding
        self.conv = nn.ConvTranspose2d(n_inputs, n_outputs, kernel_size, stride=stride, padding=(padding[0], 0), dilation=dilation)
        self.dropout = nn.Dropout(dropout)
        self.net = nn.Sequential(self.conv, self.dropout)
        self.residualmask = nn.Conv2d(n_outputs, n_outputs, (1,1))
        self.residualnorm = GlobalLayerNorm(n_outputs, time=False)
        self.sigmoid = nn.Sigmoid()
        self.residual = nn.Conv2d(n_outputs, n_outputs, (1,1))
        self.relu = nn.ReLU()
        self.norm = GlobalLayerNorm(n_outputs, time=False)

    def forward(self, x, res=None):
        #x: [B, C, F, T]
        #return: [B, C', F, T]
        T = x.shape[-1]
        out = self.net(x) [..., -T:]
        out = self.activation(out)
        out = self.norm(out)
        if res is not None:
            B, C, F, T = res.shape
            if F > out.shape[-2]:
                pad = torch.zeros((B, C, F-out.shape[-2], T), device=out.device, dtype=out.dtype)
                out = torch.cat([out, pad], dim=-2)
            elif F < out.shape[-2]:
                out = out[:, :, :F]
            
            mask = self.sigmoid(self.residualnorm(self.residualmask(res)))
            out = mask * self.relu(self.residual(res)) + (1.0-mask) * out
        return out
    
    def reset(self):
        self.norm.reset()
        self.residualnorm.reset()




class GTSA(nn.Module):
    '''
    import torch
    from modules import GTSA
    transformer = GTSA(3, 201, 500)
    x = torch.rand(3, 201, 6)
    y = transformer(x)
    print(x.shape)
    '''
    def __init__(self, num_mics, num_freqs, segment_length, num_layers, num_heads, model_dim, fn_dim, 
                maxlen=500, dropout=0.0, sample_rate=16000, win_length=25, hop_length=10, n_fft=400):
        super(GTSA, self).__init__()
        self.segment_length = segment_length
        self.num_freqs = num_freqs
        self.stft = STFT(sample_rate=sample_rate, win_length=win_length, 
                hop_length=hop_length, n_fft=n_fft)

        self.istft = ISTFT(sample_rate=sample_rate, win_length=win_length, 
                        hop_length=hop_length, n_fft=n_fft)
        
        # self.trans1 = TemporalConv2d(num_mics, 32, (5,32), 1, 1, (2,31), dropout=dropout)
        # self.relu = nn.ReLU()
        # self.trans2 = TemporalConv2d(32, 1, (5,32), 1, 1, (2,31), dropout=dropout)
        # self.norm = GlobalLayerNorm(num_freqs)
        # self.trans3 = TemporalConv2d(1, 2, (5,32), 1, 1, (2,31), dropout=dropout)
        num_channels = [8, 16, 32, 32]
        kernel_size = 3
        convs = []
        deconvs = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = (2 * num_mics - 1)if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            convs += [TemporalConv2d(in_channels, out_channels, (5,kernel_size), stride=(2,1), dilation=(1,dilation_size),
                                     padding=(2,(kernel_size-1) * dilation_size), dropout=dropout)]
            
            dilation_size = 2 ** (num_levels-i-1)
            if i==0:
                deconvs = [TemporalConvTranspose2d(out_channels, 2, (5,kernel_size), stride=(2,1), dilation=(1,dilation_size),
                            padding=(2,(kernel_size-1) * dilation_size), dropout=dropout)]
            else:
                deconvs = [TemporalConvTranspose2d(out_channels, in_channels, (5,kernel_size), stride=(2,1), dilation=(1,dilation_size),
                            padding=(2,(kernel_size-1) * dilation_size), dropout=dropout)] + deconvs
        
        self.convlist = nn.ModuleList(convs)
        self.deconvlist = nn.ModuleList(deconvs)


        transformer = []
        for i in range(num_layers):
            #transformer += [TransformerLayer(num_heads, 2*model_dim, fn_dim, maxlen, dropout)]
            transformer += [TransformerLayer(num_heads, (num_freqs // 16 + 1)*num_channels[-1], fn_dim, maxlen, dropout)]
        self.layers = nn.ModuleList(transformer)
        

    def forward(self, x):
        # x = B x C x F x T x 2
        # #[B, C, F, T, 2]
        # B, C, F, T, _ = x.shape
        # noisy = x[:,0]
        # x = torch.sqrt(x[..., 0]**2 + x[..., 1]**2 + 1e-10)
        # #->[B, F, T]
        # x = self.trans1(x)
        # x = self.relu(x)
        # x = self.trans2(x).squeeze(1)
        # #[B, F, T]
        # x = torch.cat(torch.split(x, 1, dim=1), dim=2).squeeze(1)
        #[B, 2F, T]
        #x = self.norm(x)
        noisy = x[:,0]
        angle = torch.arctan(x[..., 1]/(x[..., 0] + EPS) + EPS)
        angle = angle[:, 0].unsqueeze(1) - angle[:, 1:]
        mag = torch.sqrt(x[..., 0]**2 + x[..., 1]**2 + 1e-10)
        x = torch.cat([mag, angle], dim = 1)

        #[B, 2M-1, F, T]
        residuals = [x]
        index = 0
        for m in self.convlist:                
            x = m(x)
            residuals += [x]
            index += 1
        
        B, C, F, T = x.shape
        #[B, C, F, T] -> [B, C*F, T]
        x = x.reshape(B, C*F, T).contiguous()
        for m in self.layers:
            x = m(x)
        x = x.reshape(B, C, F, T)
        # # [B, 2, F, T]
        # x = self.trans3(x.unsqueeze(1))
        # x = torch.stack(torch.split(x, F, dim=1), dim=-1)

        B, C, F, T = x.shape
        index = -2
        for m in self.deconvlist[:-1]:
            x = m(x, residuals[index])
            index -= 1
        #last one, [B, 2, F, T] -> [B, F, T, 2]
        x = self.deconvlist[-1](x).permute(0, 2, 3, 1)
        
        x = decompress_cIRM(x)
        enhanced_real = x[...,0] * noisy[...,0] - x[...,1] * noisy[...,1]
        enhanced_imag = x[...,1] * noisy[...,0] + x[...,0] * noisy[...,1]
        #[B, F, T, 2]
        x = torch.stack([enhanced_real, enhanced_imag], dim=-1)
        return x
    
    def reset(self):
        # self.trans1.reset()
        # self.trans2.reset()
        for m in self.convlist:
            m.reset()
        for m in self.deconvlist:
            m.reset()
        for m in self.layers:
            m.reset()
        #self.norm.reset()

    
    def stft_trans(self, x):
        #B*N, M, K
        B, M, L = x.shape
        #B*N*M, L
        x = x.reshape(-1,L)
        #B*N, M, F, T, 2
        x = self.stft(x).reshape(B,M,-1,self.num_freqs,2).transpose(2,3)
        return x
    
    def istft_trans(self, x):
        B, F, T, _ = x.shape
        #B*N, T, F, 2
        x = x.permute(0,2,1,3)
        #B*N, K 
        x = self.istft(x)
        return x
    
    def segmentation(self, x):
        x, gap = segmentation(x, self.segment_length)
        return x, gap
    
    def overadd(self, x, gap):
        #B, N, K
        x = over_add(x, gap)
        return x

    
    def preprocessing(self, mixture):
        """
        mixture, source, noise: B, M, L
        output: N, B, M, F, T, 2
        """
        batch_size = len(mixture)
        #B*N, M, K
        seg_x, gap = self.segmentation(mixture)
        #B*N, M, F, T, 2
        x = self.stft_trans(seg_x)
        #N, B, M, F, T, 2
        x = x.reshape([batch_size, -1]+[*x.shape[1:]]).transpose(0,1)
        return x, gap
    
    def postprocessing(self, sp, gap):
        """
        sp: N, B, F, T, 2
        """
        N, B, F, T, _ = sp.shape
        #N*B, F, T, 2
        sp = sp.reshape(N*B, F, T, 2)
        #N*B, K
        sp = self.istft_trans(sp)
        sp = sp.reshape(N, B, -1).permute(1,0,2)
        pred_s = self.overadd(sp, gap)
        return pred_s
       
    
    def realtime_process(self, mixture, flag=False):
        B, C, T = mixture.shape
        if not flag:
            pad = torch.zeros((B, C, self.segment_length // 2), dtype=mixture.dtype, device=mixture.device)
            mixture = torch.cat([pad, mixture], dim=-1)
        x, gap = self.preprocessing(mixture)
        #N, B, C, F, T, 2
        N, B, C, F, T, _ = x.shape
        if not flag:
            self.reset()
        
        pred_source = torch.zeros((N, B, F, T, 2), dtype=x.dtype, device=x.device)
        #[N, B, C, F, T]
        for idx in range(N):
            inp = x[idx]
            #with autocast():
            preds = self.forward(inp)
            pred_source[idx] = preds
        
        # #[B, F, N*T]
        # pred_source = torch.cat(torch.split(pred_source, 1, dim=0), dim=-2).squeeze(0)
        # noisy = torch.cat(torch.split(x, 1, dim=0), dim=-2)
        # speech_mask = ComplexTensor(pred_source[..., 0], pred_source[..., 1])
        # noise_mask = ComplexTensor(pred_source[..., 2], pred_source[..., 3])
        # noisy = ComplexTensor(noisy[..., 0], noisy[..., 1]).squeeze(0)

        # #[B, F, N*T]
        # pred_source = self.beamformer(speech_mask, noise_mask, noisy)
        # #[B, F, N*T, 2]
        # pred_source = torch.stack([pred_source.real, pred_source.imag], dim=-1)
        # pred_source = torch.stack(torch.split(pred_source, T, dim=-2), dim=0)


        #[N, B, F, T, 2] -> [B, L]
        pred_source = self.postprocessing(pred_source, gap)
        if not flag:
            pred_source = pred_source[..., self.segment_length // 2: ]
        return pred_source


    
    def compute_loss(self, source, pred_source, length):
        #source: B, L
        # sf = self.stft(source)
        # spf = self.stft(pred_source)
        # m = torch.sqrt(sf[...,0]**2 + sf[...,1]**2).unsqueeze(-1)
        # pm = torch.sqrt(spf[...,0]**2 + spf[...,1]**2).unsqueeze(-1)
        # mphase = sf / (m + EPS)
        # pmphase = spf / (pm + EPS)
        # m = m**(0.3)
        # pm = pm**(0.3)
        # mae = 0.7*torch.mean(torch.abs(m-pm)) + 0.3*torch.mean(torch.abs(m*mphase - pm*pmphase))

        # #实际上是stoi
        mae = stoi_loss(source, pred_source, length)

        sisnr = -cal_si_snr(pred_source, source, length) 
        loss = 0.7 * mae + 0.3 * sisnr
        print(sisnr)
        if torch.isnan(loss):
            mae = mae.fill_(0.0)
            sisnr = sisnr.fill_(0.0)
            loss = loss.fill_(0.0)
        return loss, mae, sisnr


if __name__ == "__main__":
    import time

    with torch.no_grad():
        model = GTSA(num_mics=3, num_freqs=201, segment_length=3200, num_layers=4, num_heads=4, model_dim=201, fn_dim=1024, 
                     maxlen=500, dropout=0.0, sample_rate=16000, win_length=25, hop_length=10, n_fft=400)

        ipt = torch.rand(1, 3, 64000)  # 1.6s
        ipt_len = ipt.shape[-1]
        # 1000 frames (16s) - 5.65s (35.31%，纯模型) - 5.78s
        # 500 frames (8s) - 3.05s (38.12%，纯模型) - 3.04s
        # 200 frames (3.2s) - 1.19s (37.19%，纯模型) - 1.20s
        # 100 frames (1.6s) - 0.62s (38.75%，纯模型) - 0.65s
        start = time.time()
        
        # complex_tensor, _,  gap = model.preprocessing(ipt)
        # print(complex_tensor.shape)
        # print(f"STFT: {time.time() - start}, {complex_tensor.shape}")
        
        
        # enhanced_complex_tensor, f, s = model(complex_tensor[0], f, s)
        # enhanced_complex_tensor = enhanced_complex_tensor.detach().permute(0, 2, 3, 1)
        # print(complex_tensor.shape, enhanced_complex_tensor.shape)
        # print(f"Model Inference: {time.time() - start}")
        
        # enhanced = model.postprocessing(torch.stack([enhanced_complex_tensor]*len(complex_tensor),dim=0), gap)
        # print(f"iSTFT: {time.time() - start}")
        
        pred_source = model.realtime_process(ipt)
        print(f"Real Time Process for {ipt.shape[-1]/16000}s: {time.time() - start}")
        print(pred_source.shape)
        
        parameters = sum(param.numel() for param in model.parameters())
        print(str(parameters / 10**6) +' MB ')