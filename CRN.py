import torch
from torch.nn import functional
from torch import nn
from torch.cuda.amp import autocast as autocast
import torch.nn.init as init
from torch_complex import ComplexTensor
import torch_complex
from utility import *
from speechbrain.processing.features import STFT,ISTFT

EPS = 1e-8

# class MutiheadAttention(nn.Module):
#     '''
#     import torch
#     from modules import MutiheadAttention
#     attention = MutiheadAttention(3, 201, 500, batch_size=3)
#     x = torch.rand(3, 6, 201)
#     y = attention(x)
#     print(x.shape, attention.bk[0,:,0], attention.bv[0,:,0])
#     '''
#     def __init__(self, num_heads, model_dim, maxlen, batch_size = 1):
#         super(MutiheadAttention, self).__init__()
#         self.num_heads = num_heads
#         self.model_dim = model_dim
#         self.maxlen = maxlen
#         self.batch_size = batch_size
        
#         self.ql = nn.Linear(model_dim, model_dim)
#         self.kl = nn.Linear(model_dim, model_dim)
#         self.vl = nn.Linear(model_dim, model_dim)
#         self.linear = nn.Linear(model_dim, model_dim)
#         self.softmax = nn.Softmax(dim=-1)
        
#         self.delta = nn.Parameter(torch.ones(1))
#         self.d = torch.sqrt(torch.tensor(self.model_dim).type(torch.float32))


#         self.register_buffer('bk', torch.zeros((batch_size * num_heads, maxlen, model_dim // num_heads)))
#         self.register_buffer('bv', torch.zeros((batch_size * num_heads, maxlen, model_dim // num_heads)))


#     def forward(self, x):
#         # x = B x T x F
#         B, T, F = x.shape
#         q = torch.cat(torch.split(self.ql(x), F // self.num_heads, dim=-1), dim=0) 
#         k = torch.cat(torch.split(self.kl(x), F // self.num_heads, dim=-1), dim=0)
#         v = torch.cat(torch.split(self.vl(x), F // self.num_heads, dim=-1), dim=0)
        
#         k = torch.cat([self.bk[:, T:], k], dim=1)
#         v = torch.cat([self.bv[:, T:], v], dim=1)
#         ind = torch.arange(1, self.maxlen+1).unsqueeze(1).repeat(1,self.maxlen).to(x.device)
#         G = torch.exp(-(ind-ind.transpose(0,1))**2 / (self.delta**2 + EPS)).unsqueeze(0)
#         #[E*B, T, F/E] * [E*B, F/E, maxlen]
#         x = torch.matmul(q, k.transpose(1,2)) * G[:, -T:] / self.d
#         #[E*B, T, maxlen] * [E*B, maxlen, F/E]
#         x = torch.matmul(self.softmax(x), v)

#         x = torch.cat(torch.split(x, B, dim=0), dim=-1)
#         x = self.linear(x)

#         with torch.no_grad():
#             self.bk = k.clone()
#             self.bv = v.clone()
#         return x
    
#     def reset(self):
#         self.bk = torch.zeros_like(self.bk)
#         self.bv = torch.zeros_like(self.bv)


# class TransformerLayer(nn.Module):
#     '''
#     import torch
#     from modules import TransformerLayer
#     transformer = TransformerLayer(3, 201, 500)
#     x = torch.rand(3, 201, 6)
#     y = transformer(x)
#     print(x.shape)
#     '''
#     def __init__(self, num_heads, model_dim, fn_dim, maxlen=500, dropout=0.0):
#         super(TransformerLayer, self).__init__()
#         self.attention = MutiheadAttention(num_heads, model_dim, maxlen)
#         self.norm_a =  GlobalLayerNorm(model_dim, last=True, time=False)
#         self.activation = nn.ReLU()
#         self.linear_in = nn.Linear(model_dim, fn_dim)
#         self.linear_out = nn.Linear(fn_dim, model_dim)
#         self.norm_i = GlobalLayerNorm(model_dim, last=True, time=False)
#         self.dropout = nn.Dropout(dropout)


#     def forward(self, x):
#         # x = B x F x T
#         #[B, T, F]
#         x = x.transpose(1,2)
#         x = self.attention(x) + x
#         x = self.norm_a(x)
#         res = x
#         x = self.activation(self.linear_in(x))
#         x = self.dropout(x)
#         x = self.linear_out(x) + res
#         x = self.norm_i(x).transpose(1,2)
#         return x
    
#     def reset(self):
#         self.attention.reset()
#         self.norm_a.reset()
#         self.norm_i.reset()


class GlobalLayerNorm(nn.Module):
    '''
    import torch
    from modules import GlobalLayerNorm
    norm = GlobalLayerNorm(201)
    x = torch.rand(3, 201, 6)
    y = norm(x)
    print(x.shape, norm.step)
    '''
    def __init__(self, dim, last = False, time=True):
        super(GlobalLayerNorm, self).__init__()
        self.mean = None
        self.var = None
        self.step = 0
        self.time = time
        
        if last:
            self.weight = nn.Parameter(torch.ones(1, 1, 1, dim))
            self.bias = nn.Parameter(torch.zeros(1, 1, 1, dim))
        else:
            self.weight = nn.Parameter(torch.ones(1, dim, 1, 1))
            self.bias = nn.Parameter(torch.zeros(1, dim, 1, 1))


    def forward(self, x):
        # x = B x F x T
        T = x.shape[-1]
        mean = torch.mean(x, (1,2,3), keepdim=True)
        var = torch.mean((x-mean)**2, (1,2,3), keepdim=True)

        if self.mean is None or not self.time:
            global_mean = mean
            global_var = var
        else:
            alpha = self.step / (self.step + T)
            global_mean = alpha * self.mean + (1.0 - alpha) * mean
            global_var = alpha * (self.var + (global_mean - self.mean) ** 2) + (1.0 - alpha) * (var + (global_mean - mean) ** 2)

        x = (x - global_mean) / (torch.sqrt(global_var+EPS) + EPS) * self.weight + self.bias
        
        self.step += T
        self.mean = global_mean.detach()
        self.var = global_var.detach()
        return x
    
    def reset(self):
        self.mean = None
        self.var = None
        self.step = 0

class Linear_T(nn.Linear):
    '''
    import torch
    from modules import Linear_T
    linear = Linear_T(20, 5)
    x = torch.rand(3, 10, 6)
    y = linear(x)
    print(x.shape, linear.step)
    '''
    def __init__(self, in_features, out_features, bias=True):
        super().__init__(in_features, out_features, bias)
        self.out = None
        self.step = 0
    
    def forward(self, x):
        T = x.shape[-1]
        if self.step + T > self.in_features:
            last = self.step + T - self.in_features
            weight = torch.cat([self.weight[:, self.step:], self.weight[:, :last]], dim=1)
        else:
            weight = self.weight[..., self.step:self.step+T]

        if self.out is None:
            out = functional.linear(x , weight, self.bias)
        else:
            out = self.out + functional.linear(x, weight)
        
        self.out = out.clone().detach()
        self.step = (self.step+T) % self.in_features
        return out
    
    def reset(self):
        self.out = None
        self.step = 0

class SequenceModel(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, num_layers, bidirectional, linear=True,
                 timeasfeature = False, sequence_model="GRU",output_activate_function="Tanh"):
        """
        ---------------------------------------
        import torch
        from modules import SequenceModel
        model = SequenceModel(20, 10, 10, 2, bidirectional=False, timeasfeature=True)
        x = torch.rand(3, 20, 6)
        y = model(x)
        print(x.shape, model.h)
        """
        super().__init__()
        self.timeasfeature = timeasfeature
        self.model_type = sequence_model
        self.linear = linear
        if not linear:
            hidden_size = input_size

        # Sequence layer
        if sequence_model == "LSTM":
            self.sequence_model = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                bidirectional=bidirectional,
            )   
        elif sequence_model == "GRU":
            self.sequence_model = nn.GRU(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                bidirectional=bidirectional,
            )
        else:
            raise NotImplementedError(f"Not implemented {sequence_model}")

        # Fully connected layer
        if linear and bidirectional:
            self.fc_output_layer = nn.Linear(hidden_size * 2, output_size)
        elif linear:
            self.fc_output_layer = nn.Linear(hidden_size, output_size)

        self.norm = GlobalLayerNorm(output_size, last=True, time=False)
        # Activation function layer
        if output_activate_function:
            if output_activate_function == "Tanh":
                self.activate_function = nn.Tanh()
            elif output_activate_function == "ReLU":
                self.activate_function = nn.ReLU()
            elif output_activate_function == "Sigmoid":
                self.activate_function = nn.Sigmoid()
            else:
                raise NotImplementedError(f"Not implemented activation function {self.activate_function}")

        self.output_activate_function = output_activate_function
        self.h = None

    def forward(self, x):
        """
        Args:
            x: [B, F, T]
        Returns:
            [B, F, T]
        """
        assert x.dim() == 3
        self.sequence_model.flatten_parameters()

        # contiguous 使元素在内存中连续，有利于模型优化，但分配了新的空间
        # 建议在网络开始大量计算前使用一下
        x = x.permute(0, 2, 1).contiguous()  # [B, F, T] => [B, T, F]
        o, h = self.sequence_model(x, self.h)
        if self.linear:
            o = self.fc_output_layer(o)
        if self.output_activate_function:
            o = self.activate_function(o)
        o = self.norm(o.unsqueeze(1)).squeeze(1)
        o = o.permute(0, 2, 1).contiguous()  # [B, T, F] => [B, F, T]

        
        if self.model_type == "LSTM":
            self.h = (h[0].detach(), h[1].detach())
        else:
            self.h = h.detach()
        return o

    def reset(self):
        self.h = None
        if self.timeasfeature and self.linear:
            self.fc_output_layer.reset()


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

        self.buffer = None

    def forward(self, x):
        #x: [B, C, F, T]
        #return: [B, C', F, T]
        B, C, F, T = x.shape
        if self.buffer is None:
            self.buffer = torch.zeros((B, C, F, self.padding), device=x.device, dtype=x.dtype)

        buffer = self.buffer.clone()
        inp = torch.cat([buffer, x], dim=-1)
        out = self.net(inp)
        out = self.activation(out) 
        out = self.norm(out)
        if T > self.padding:
            self.buffer = x[..., -self.padding:].clone().detach()
        else:
            self.buffer[..., :self.padding - T] = self.buffer[..., T-self.padding:]
            self.buffer[..., -T:] = x.clone().detach()
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
        #
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


class TemporalCRN(nn.Module):
    """
    网络需要能够实时处理输入的语音。
    在配置上，每来一段长为1600个采样点(100ms)的语音就要进行降噪。
    网络的输入为这样的一段语音,所有的网络结构都应该在一定程度上支持实时操作，保存之前的状态(虽然效果不一定好)。

    代码最上方注释部分是实现GTSA时我按照自己想法实现的时序Transformer,实际使用后效果并不好，但可供参考。
    TemporalCNN是因果卷积模块，效果比较好，可以直接使用或者略加修改。
    GlobalLayerNorm使用滑动平均策略时效果一般没有直接进行Norm效果好,所以一般不使用滑动平均。
    SequenceModel是封装好的保存状态的LSTM或者GRU网络,效果不错，可以略加修改。
    """
    def __init__(self, num_channels, num_freqs, hidden, segment_length,
                    num_layers = 1, num_inputs=3, kernel_size=3, dropout=0.0, sample_rate=16000, 
                    win_length=25, hop_length=10, n_fft=400):
        super(TemporalCRN, self).__init__()
        self.segment_length = segment_length
        self.num_freqs = num_freqs
        self.stft = STFT(sample_rate=sample_rate, win_length=win_length, 
                hop_length=hop_length, n_fft=n_fft)

        self.istft = ISTFT(sample_rate=sample_rate, win_length=win_length, 
                        hop_length=hop_length, n_fft=n_fft)


        convs = []
        deconvs = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = (2 * num_inputs - 1) if i == 0 else num_channels[i-1]
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

        self.gru = SequenceModel((num_freqs // 16 + 1)*num_channels[-1], (num_freqs // 16 + 1)*num_channels[-1], hidden, num_layers, False, 
                                 linear=True, sequence_model="GRU",output_activate_function="ReLU")
        

    def forward(self, x):
        '''
        x: shape [B, M, F, T, 2]
        B: batch_size
        M: multichannel size
        F: frequency size
        T: frame length
        2: real part and image part of a complex number
        '''
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
        x = self.gru(x)
        #[B, C, F, T]
        x = x.reshape(B, C, F, T)

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
        for m in self.convlist:
            m.reset()
        for m in self.deconvlist:
            m.reset()
        self.gru.reset()
    
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
        '''
        传入的训练数据是16000到60000之间长度的带噪数据,并且目前只实现了batch_size=1的情况。
        mixture需要切分为3200每段,共N段,步长为1600。切分之后对每段做短时傅里叶变换。
        切分的目的是模拟真实的实时处理场景。
        进行降噪之后再转换为时域数据。
        '''
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

        #[N, B, F, T, 2] -> [B, L]
        pred_source = self.postprocessing(pred_source, gap)
        if not flag:
            pred_source = pred_source[..., self.segment_length // 2: ]
        return pred_source


    
    def compute_loss(self, source, pred_source, length):
        '''
        loss包括stoi和sisnr两部分，均是评价语音降噪效果的损失函数。
        使用mae或者mse最终降噪效果似乎都不咋样。
        '''
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

        stoi = stoi_loss(source, pred_source, length)
        sisnr = -cal_si_snr(pred_source, source, length) 
        loss = 0.7 * stoi + 0.3 * sisnr
        print(sisnr)
        if torch.isnan(loss):
            stoi = stoi.fill_(0.0)
            sisnr = sisnr.fill_(0.0)
            loss = loss.fill_(0.0)
        return loss, stoi, sisnr


if __name__ == "__main__":
    import time

    with torch.no_grad():
        model = TemporalCRN(num_channels=[16, 32, 64, 128], num_freqs=201, hidden=512, 
                            segment_length = 3200, num_layers = 2, num_inputs=3, kernel_size=3, dropout=0.0)
        ipt = torch.rand(1, 3, 32000)  # 1.6s
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
        print(str(parameters / 10**6) +' M ')