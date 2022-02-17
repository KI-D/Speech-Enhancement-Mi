import sys
sys.path.append("../")
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import Conv1d, AvgPool1d, Conv2d
import torchaudio
from torch.cuda.amp import autocast as autocast
from torch.nn.utils import weight_norm, remove_weight_norm, spectral_norm
from utility import *
from speechbrain.processing.features import STFT,ISTFT

LRELU_SLOPE = 0.1


# class TemporalConv1d(nn.Module):
#     """
#     ---------------------------------------
#     import torch
#     from hifigan import TemporalConv1d
#     model = TemporalConv1d(1, 16, 3, 1, 5, 10)
#     x = torch.rand(3, 1, 20)
#     y = model(x)
#     print(x.shape, model.buffer[0, 0, :])
#     """
#     def __init__(self, n_inputs, n_outputs, kernel_size= 1, stride= 1, dilation= 1, padding=0, bias=True):
#         super(TemporalConv1d, self).__init__()
#         self.padding = padding
        
#         #手动padding
#         self.conv = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size, stride=stride, padding=0, dilation=dilation, bias=bias))
#         self.register_buffer('buffer', None)

#     def forward(self, x):
#         #x: [B, C, T]
#         #return: [B, C', T]
#         B, C, T = x.shape
#         if self.padding == 0:
#             out = self.conv(x)
#             return out
        
#         if self.buffer is None:
#             with torch.no_grad():
#                 self.buffer = torch.zeros((B, C, self.padding), device=x.device, dtype=x.dtype)
#         x = torch.cat([self.buffer, x], dim=-1)

#         out = self.conv(x)
#         with torch.no_grad():
#             if T > self.padding:
#                 self.buffer = x[..., -self.padding:]
#             elif T == self.padding:
#                 self.buffer = x.clone()
#             else:
#                 self.buffer[..., :self.padding - T] = self.buffer[..., T - self.padding:]
#                 self.buffer[..., -T:] = x
#         return out
    
#     def reset(self):
#         self.buffer = None
    
#     def remove_weight_norm(self):
#         remove_weight_norm(self.conv)


# class TemporalConvTranspose1d(nn.Module):
#     """
#     ---------------------------------------
#     import torch
#     from modules import TemporalConvTranspose1d
#     model = TemporalConvTranspose2d(16, 1, 3, 1, 5, 10)
#     x = torch.rand(3, 16, 20, 20)
#     res = torch.rand(3, 1, 20, 20)
#     y = model(x, res)
#     """
#     def __init__(self, n_inputs, n_outputs, kernel_size = 1, stride = 1, dilation = 1, padding = 0, bias=True):
#         super(TemporalConvTranspose1d, self).__init__()
#         self.padding = padding
#         #手动padding
#         self.conv = weight_norm(nn.ConvTranspose1d(n_inputs, n_outputs, kernel_size, stride=stride, padding=0, dilation=dilation, bias=bias))


#     def forward(self, x, inital_T):
#         #x: [B, C, T]
#         #return: [B, C', T]
#         out = self.conv(x) [..., -inital_T:]
#         return out
    
#     def remove_weight_norm(self):
#         remove_weight_norm(self.conv)




# class ResBlock(torch.nn.Module):
#     def __init__(self, in_channel, out_channel, skip_out_channels, kernel_size=3, dilation=1, bias=True, dropout=0.0):
#         super(ResBlock, self).__init__()
#         self.dropout = nn.Dropout(dropout)
#         self.conv = TemporalConv1d(in_channel, out_channel, kernel_size, 1, dilation=dilation, padding=(kernel_size-1) * dilation)

#         self.conv_res = TemporalConv1d(in_channel, out_channel, bias=bias)
#         self.conv_out = TemporalConv1d(out_channel, out_channel, bias=bias)
#         self.conv_skip = TemporalConv1d(out_channel, skip_out_channels, bias=bias)
#         # if activation == 'ReLU':
#         #     self.activation = nn.ReLU()
#         # elif activation == 'LeakyReLU':
#         #     self.activation = nn.LeakyReLU(LRELU_SLOPE)
#         # elif activation == 'PReLU':
#         #     self.activation = nn.PReLU()


#     def forward(self, x):
#         residual = x
#         x = self.dropout(x)
#         x = self.conv(x)
#         x = torch.tanh(x) * torch.sigmoid(x)
#         skip = self.conv_skip(x)
#         out = self.conv_res(residual) + self.conv_out(x)
#         return out, skip

#     def remove_weight_norm(self):
#         self.conv.remove_weight_norm()
#         self.conv_res.remove_weight_norm()
#         self.conv_out.remove_weight_norm()
#         self.conv_skip.remove_weight_norm()

#     def reset(self):
#         self.conv.reset()
#         self.conv_res.reset()
#         self.conv_out.reset()
#         self.conv_skip.reset()


# class Generator(torch.nn.Module):
#     def __init__(self, input_channel, channel, num_layers=10, num_stacks = 2, kernel_size = 3, post_channel = 128, post_layers = 12):
#         super(Generator, self).__init__()
#         self.post = True
#         self.conv_pre = TemporalConv1d(input_channel, channel, 16, 1, 1, 15)
#         self.num_layers = num_layers
#         net = []
#         for _ in range(num_stacks):
#             dilation = 1
#             for j in range(self.num_layers):
#                 inputs = channel if j==0 else channel
#                 net += [ResBlock(inputs, channel, channel, kernel_size, dilation)]
#                 dilation *= 2
#         self.net = nn.ModuleList(net)
#         self.conv_post = TemporalConv1d(channel, 1, 16, 1, 1, 15)

#         self.post_layers = post_layers
#         postnet = []
#         postnet += [TemporalConv1d(1, post_channel)]
#         for _ in range(post_layers-2):
#             postnet += [TemporalConv1d(post_channel, post_channel)]
#         postnet += [TemporalConv1d(post_channel, 1)]
#         self.postnet = nn.ModuleList(postnet)


#     def forward(self, x, post = True):
#         self.post = post
#         B, _, T = x.shape
#         x = torch.tanh(self.conv_pre(x))
#         skip = 0
#         for n in self.net:
#             x, s = n(x)
#             skip += s
#         x = skip / self.num_layers
#         x = torch.tanh(self.conv_post(x))
            
#         before = x
#         if self.post:
#             for p in self.postnet:
#                 x = torch.tanh(p(x))
#         return x, before

#     def reset(self):
#         self.conv_pre.reset()
#         self.conv_post.reset()
#         for c in self.net:
#             c.reset()
#         if self.post:
#             for c in self.postnet:
#                 c.reset()
    
#     def remove_weight_norm(self):
#         for c in self.net:
#             c.remove_weight_norm()
#         for c in self.postnet:
#             c.remove_weight_norm()
#         self.conv_pre.remove_weight_norm()
#         self.conv_post.remove_weight_norm()


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
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.0, activation="Tanh"):
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
        self.conv = weight_norm(nn.Conv2d(n_inputs, n_outputs, kernel_size, stride=stride, padding=(padding[0], 0), dilation=dilation))
        self.dropout = nn.Dropout(dropout)
        self.net = nn.Sequential(self.conv, self.dropout)

        self.buffer = None

    def forward(self, x):
        #x: [B, C, F, T]
        #return: [B, C', F, T]
        B, C, F, T = x.shape
        if self.padding > 0:
            if self.buffer is None:
                self.buffer = torch.zeros((B, C, F, self.padding), device=x.device, dtype=x.dtype)
            buffer = self.buffer.clone()
            x = torch.cat([buffer, x], dim=-1)

        out = self.net(x)
        out = self.activation(out) * torch.sigmoid(out)

        if self.padding > 0:
            if T > self.padding:
                self.buffer = x[..., -self.padding:].clone().detach()
            elif T == self.padding:
                self.buffer = x.clone().detach()
            else:
                self.buffer[..., :self.padding - T] = self.buffer[..., T-self.padding:]
                self.buffer[..., -T:] = x.clone().detach()
        return out
    
    def reset(self):
        self.buffer = None


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
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.0, activation="Tanh"):
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
        self.conv = weight_norm(nn.ConvTranspose2d(n_inputs, n_outputs, kernel_size, stride=stride, padding=(padding[0], 0), dilation=dilation))
        self.dropout = nn.Dropout(dropout)
        self.net = nn.Sequential(self.conv, self.dropout)
        self.residualmask = weight_norm(nn.Conv2d(n_outputs, n_outputs, (1,1)))
        self.sigmoid = nn.Sigmoid()
        self.residual = weight_norm(nn.Conv2d(n_outputs, n_outputs, (1,1)))
        self.tanh = nn.Tanh()

    def forward(self, x, res=None):
        #x: [B, C, F, T]
        #return: [B, C', F, T]
        T = x.shape[-1]
        out = self.net(x) [..., -T:]
        out = self.activation(out) * self.sigmoid(out)

        if res is not None:
            B, C, F, T = res.shape
            if F > out.shape[-2]:
                pad = torch.zeros((B, C, F-out.shape[-2], T), device=out.device, dtype=out.dtype)
                out = torch.cat([out, pad], dim=-2)
            elif F < out.shape[-2]:
                out = out[:, :, :F]
            
            mask = self.sigmoid(self.residualmask(res))
            out = mask * self.tanh(self.residual(res)) + (1.0-mask) * out
        return out
    
    def reset(self):
        pass



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

        if self.mean is None or not self.time:
            global_mean = mean
        else:
            alpha = self.step / (self.step + T)
            global_mean = alpha * self.mean + (1.0 - alpha) * mean

        x = (x - global_mean) * self.weight + self.bias
        
        self.step += T
        self.mean = global_mean.detach()
        return x
    
    def reset(self):
        self.mean = None
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
            self.fc_output_layer = weight_norm(nn.Linear(hidden_size * 2, output_size))
        elif linear:
            self.fc_output_layer = weight_norm(nn.Linear(hidden_size, output_size))

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
        self.norm = GlobalLayerNorm(output_size, last=True, time=True)
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


class Generator(nn.Module):
    """
    ---------------------------------------
    import torch
    from modules import TemporalCRN
    model = TemporalCRN([4,8,16,32,64], 201, 128, 256, 400)
    x = torch.rand(3, 3, 201, 6, 2)
    y = model(x)
    """
    def __init__(self, num_channels, num_freqs, hidden, segment_length,
                    num_layers = 1, num_inputs=3, kernel_size=3, dropout=0.0, sample_rate=16000, 
                    win_length=25, hop_length=10, n_fft=400):
        super(Generator, self).__init__()
        self.segment_length = segment_length
        self.num_freqs = num_freqs
        self.stft = STFT(sample_rate=sample_rate, win_length=win_length, 
                hop_length=hop_length, n_fft=n_fft)

        self.istft = ISTFT(sample_rate=sample_rate, win_length=win_length, 
                        hop_length=hop_length, n_fft=n_fft)

        self.post = True
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
                                 linear=True, sequence_model="LSTM",output_activate_function="Tanh")
        
        self.post_layers = 12
        post_channel = 128
        postnet = []
        postnet += [TemporalConv2d(2, post_channel, (1,1), (1,1), (1,1), (0,0))]
        for _ in range(10):
            postnet += [TemporalConv2d(post_channel, post_channel, (1,1), (1,1), (1,1), (0,0))]
        postnet += [TemporalConv2d(post_channel, 2, (1,1), (1,1), (1,1), (0,0))]
        self.postnet = nn.ModuleList(postnet)
        

    def forward(self, x, post = True):
        self.post = post
        #[B, M, F, T, 2]
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
        x = self.deconvlist[-1](x)
        x_before = None
        if post:
            x_before = x.permute(0, 2, 3, 1)
            x_before = decompress_cIRM(x_before)
            enhanced_real = x_before[...,0] * noisy[...,0] - x_before[...,1] * noisy[...,1]
            enhanced_imag = x_before[...,1] * noisy[...,0] + x_before[...,0] * noisy[...,1]
            #[B, F, T, 2]
            x_before = torch.stack([enhanced_real, enhanced_imag], dim=-1)

            for m in self.postnet:
                x = m(x)
        #last one, [B, 2, F, T] -> [B, F, T, 2]
        x = x.permute(0, 2, 3, 1)

        x = decompress_cIRM(x)
        enhanced_real = x[...,0] * noisy[...,0] - x[...,1] * noisy[...,1]
        enhanced_imag = x[...,1] * noisy[...,0] + x[...,0] * noisy[...,1]
        #[B, F, T, 2]
        x = torch.stack([enhanced_real, enhanced_imag], dim=-1)
        return x, x_before
    
    def reset(self):
        for m in self.convlist:
            m.reset()
        for m in self.deconvlist:
            m.reset()
        self.gru.reset()
        if self.post:
            for m in self.postnet:
                m.reset()
        
    
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
       
    
    def realtime_process(self, mixture, post = True, reset = False):
        B, C, T = mixture.shape
        if reset:
            pad = torch.zeros((B, C, self.segment_length // 2), dtype=mixture.dtype, device=mixture.device)
            mixture = torch.cat([pad, mixture], dim=-1)
        x, gap = self.preprocessing(mixture)
        #N, B, C, F, T, 2
        N, B, C, F, T, _ = x.shape
        if reset:
            self.reset()
        
        pred_source = torch.zeros((N, B, F, T, 2), dtype=x.dtype, device=x.device)
        pred_source_before = torch.zeros((N, B, F, T, 2), dtype=x.dtype, device=x.device)
        #[N, B, C, F, T]
        for idx in range(N):
            inp = x[idx]
            #with autocast():
            preds, preds_before = self.forward(inp, post)
            pred_source[idx] = preds
            pred_source_before[idx] = preds_before
        
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
        pred_source_before = self.postprocessing(pred_source_before, gap)
        if reset:
            pred_source = pred_source[..., self.segment_length // 2: ]
            pred_source_before = pred_source_before[..., self.segment_length // 2: ]
        return pred_source, pred_source_before



class DiscriminatorM(torch.nn.Module):
    def __init__(self, use_spectral_norm=False):
        super(DiscriminatorM, self).__init__()
        norm_f = weight_norm if use_spectral_norm == False else spectral_norm
        self.convs = nn.ModuleList([
            norm_f(Conv2d(1, 32, (3,3), (1,1), (1,1))),
            norm_f(Conv2d(32, 32, (3,9), (1,1), (1,4))),
            norm_f(Conv2d(32, 32, (3,8), (1,2), (1,3))),
            norm_f(Conv2d(32, 32, (3,8), (1,2), (1,3))),
            norm_f(Conv2d(32, 32, (3,6), (1,2), (1,2))),
        ])
        self.conv_post = norm_f(nn.Conv2d(32, 1, (32, 5), (32, 1), (0, 2)))
        self.pool = nn.AvgPool2d((1, 2))

    def forward(self, x):
        fmap = []
        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = self.pool(x)
        x = torch.flatten(x, 1, -1)
        return x, fmap


class MultiMelDiscriminator(torch.nn.Module):
    def __init__(self, sample_rate, nffts, n_mels):
        super(MultiMelDiscriminator, self).__init__()
        self.meltransform = nn.ModuleList([torchaudio.transforms.MelSpectrogram(sample_rate, nfft, nfft, nfft // 2, n_mels=n_mels) for nfft in nffts])
        self.discriminators = nn.ModuleList([DiscriminatorM() for _ in range(len(nffts))])
        self.pool = nn.AvgPool2d((1, 2))


    def forward(self, y_hat, y):
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        for t, d in zip(self.meltransform, self.discriminators):
            melspec = t(y)
            melspec_hat = t(y_hat)
            y_d_r, fmap_r = d(melspec)
            y_d_g, fmap_g = d(melspec_hat)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


class DiscriminatorS(torch.nn.Module):
    def __init__(self, use_spectral_norm=False):
        super(DiscriminatorS, self).__init__()
        norm_f = weight_norm if use_spectral_norm == False else spectral_norm
        self.convs = nn.ModuleList([
            norm_f(Conv1d(1, 128, 15, 1, padding=7)),
            norm_f(Conv1d(128, 128, 41, 2, groups=4, padding=20)),
            norm_f(Conv1d(128, 256, 41, 2, groups=16, padding=20)),
            norm_f(Conv1d(256, 512, 41, 4, groups=16, padding=20)),
            norm_f(Conv1d(512, 1024, 41, 4, groups=16, padding=20)),
            norm_f(Conv1d(1024, 1024, 41, 1, groups=16, padding=20)),
            norm_f(Conv1d(1024, 1024, 5, 1, padding=2)),
        ])
        self.conv_post = norm_f(Conv1d(1024, 1, 3, 1, padding=1))

    def forward(self, x):
        fmap = []
        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)
        return x, fmap


class MultiScaleDiscriminator(torch.nn.Module):
    def __init__(self):
        super(MultiScaleDiscriminator, self).__init__()
        self.discriminators = nn.ModuleList([
            DiscriminatorS(use_spectral_norm=True),
            DiscriminatorS(),
            DiscriminatorS(),
        ])
        self.meanpools = nn.ModuleList([
            AvgPool1d(4, 2, padding=2),
            AvgPool1d(4, 2, padding=2)
        ])

    def forward(self, y_hat, y):
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        for i, d in enumerate(self.discriminators):
            if i != 0:
                y = self.meanpools[i-1](y)
                y_hat = self.meanpools[i-1](y_hat)
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)
        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


# class Hifi_GAN(torch.nn.Module):
#     def __init__(self, input_channel, channel, sample_rate, nffts, n_mels, num_layers=10, num_stacks = 2, kernel_size = 3, post_channel = 128, post_layers = 12):
#         super(Hifi_GAN, self).__init__()
#         self.generator = Generator(input_channel, channel, num_layers, num_stacks, kernel_size, post_channel, post_layers)
#         self.discriminator = nn.ModuleList([MultiMelDiscriminator(sample_rate, nffts, n_mels), MultiScaleDiscriminator()])
    
#     def forward(self, x, post=True):
#         y, _ = self.generator(x)
#         return y
    
#     def reset(self):
#         self.generator.reset()
    
#     def remove_weight_norm(self):
#         print('Removing weight norm...')
#         self.generator.remove_weight_norm()


#     def discriminator_forward(self, y_hat, y):
#         r_out = []
#         g_out = []
#         fmap_r = []
#         fmap_g = []
#         for i in range(2):
#             y_d_rs, y_d_gs, fmap_rs, fmap_gs = self.discriminator[i](y_hat, y)
#             r_out += y_d_rs
#             g_out += y_d_gs
#             fmap_r += fmap_rs
#             fmap_g += fmap_gs
#         return r_out, g_out, fmap_r, fmap_g
    

#     def train_stage(self, x, y, stage = 1, y_hat = None, y_before = None, reset=True):
#         loss = 0.0
#         mode = "G"
#         if reset:
#             self.reset()
        
#         if y_hat is None:
#             mode = "D"
#             y_hat, y_before = self.generator(x, post=True)

#         if stage == 1:
#             loss = self.stft_loss(y_hat, y)
#             return loss
#         elif stage == 2:
#             loss += 0.5 * self.stft_loss(y_hat, y_before)
#             loss += 0.5 * self.stft_loss(y_hat, y)
#             return loss
#         else:
#             if mode == "D":
#                 # Discriminator
#                 r_out, g_out, fmap_r, fmap_g = self.discriminator_forward(y_hat.detach(), y)
#                 loss += self.feature_loss(fmap_r, fmap_g)
#                 loss += self.discriminator_loss(r_out, g_out)
#                 return loss, y_hat, y_before
#             if mode == "G":
#                 # Generator
#                 r_out, g_out, fmap_r, fmap_g = self.discriminator_forward(y_hat, y)
#                 loss += 0.5 * self.stft_loss(y_before, y_hat) + 0.5 * self.stft_loss(y, y_hat)
#                 loss += self.generator_loss(g_out)
#                 return loss
        

#     def feature_loss(self, fmap_r, fmap_g):
#         loss = 0
#         for dr, dg in zip(fmap_r, fmap_g):
#             for rl, gl in zip(dr, dg):
#                 loss += torch.mean(torch.abs(rl - gl))
#         return loss*2


#     def discriminator_loss(self, disc_real_outputs, disc_generated_outputs):
#         loss = 0
#         for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
#             r_loss = torch.mean((1-dr)**2)
#             g_loss = torch.mean(dg**2)
#             loss += (r_loss + g_loss)
#         return loss


#     def generator_loss(self, disc_outputs):
#         loss = 0
#         for dg in disc_outputs:
#             l = torch.mean((1-dg)**2)
#             loss += l
#         return loss
    

#     def stft_loss(self, pred, real, nfft = 400, nhop = 200, nwin = 200, window="hann_window"):
#         # if pred.dim() > 2:
#         #     pred = pred.squeeze(1)
#         # if real.dim() > 2:
#         #     real = real.squeeze(1)
        
#         # window = getattr(torch, window)(nwin).to(pred.device)
#         # pred_stft = torch.stft(pred, nfft, nhop, nwin, window)
#         # real_stft = torch.stft(real, nfft, nhop, nwin, window)
#         # pred_mag = torch.sqrt(torch.clamp(pred_stft[...,0] ** 2 + pred_stft[...,1] ** 2, min=1e-7)).transpose(2, 1)
#         # real_mag = torch.sqrt(torch.clamp(real_stft[...,0] ** 2 + real_stft[...,1] ** 2, min=1e-7)).transpose(2, 1)
#         # logmag_loss = torch.mean(torch.abs(torch.log(pred_mag) - torch.log(real_mag)))
#         # spectralconverage_loss = torch.mean(torch.norm(pred_mag - real_mag, p="fro") / torch.norm(pred_mag, p="fro"))
#         # loss = logmag_loss + spectralconverage_loss

#         if pred.dim() > 2:
#             pred = pred.squeeze(1)
#         if real.dim() > 2:
#             real = real.squeeze(1)
#         loss = -cal_si_snr(pred, real)
#         return loss



class Hifi_GAN(torch.nn.Module):
    def __init__(self, nffts, n_mels, num_channels, num_freqs, hidden, segment_length,
                    num_layers = 1, num_inputs=3, kernel_size=3, dropout=0.0, sample_rate=16000, 
                    win_length=25, hop_length=10, n_fft=400):
        super(Hifi_GAN, self).__init__()
        self.generator = Generator(num_channels, num_freqs, hidden, segment_length,
                                    num_layers, num_inputs, kernel_size, dropout, sample_rate, 
                                    win_length, hop_length, n_fft)
        self.discriminator = nn.ModuleList([MultiMelDiscriminator(sample_rate, nffts, n_mels), MultiScaleDiscriminator()])
    
    def forward(self, x, post=True):
        y, _ = self.generator.realtime_process(x, reset = False, post=post)
        return y
    
    def remove_weight_norm(self):
        print('Removing weight norm...')
        self.generator.remove_weight_norm()


    def discriminator_forward(self, y_hat, y):
        r_out = []
        g_out = []
        fmap_r = []
        fmap_g = []
        for i in range(2):
            y_d_rs, y_d_gs, fmap_rs, fmap_gs = self.discriminator[i](y_hat, y)
            r_out += y_d_rs
            g_out += y_d_gs
            fmap_r += fmap_rs
            fmap_g += fmap_gs
        return r_out, g_out, fmap_r, fmap_g
    

    def train_stage(self, x, y, stage = 1, y_hat = None, y_before = None, reset=True):
        loss = 0.0
        mode = "G"
        if stage == 1:
            post = False
        else:
            post = True

        if y_hat is None:
            mode = "D"
            y_hat, y_before = self.generator.realtime_process(x, reset = reset, post=post)
        
        if stage == 1:
            loss = self.stft_loss(y_hat, y, phase=True)
            print(loss)
            return loss
        elif stage == 2:
            loss += 0.5 * self.stft_loss(y_hat, y, phase=True)
            loss += 0.5 * self.stft_loss(y_before, y, phase=True)
            print(loss)
            return loss
        else:
            if y_hat.dim() < 3:
                y_hat = y_hat.unsqueeze(1)
            if y_before.dim() < 3:
                y_before = y_before.unsqueeze(1)
            
            if mode == "D":
                # Discriminator
                r_out, g_out, _, _ = self.discriminator_forward(y_hat.detach(), y)
                loss += self.discriminator_loss(r_out, g_out)
                print(f"Discriminator loss: {loss}")
                return loss, y_hat, y_before
            if mode == "G":
                # Generator
                _, g_out, fmap_r, fmap_g = self.discriminator_forward(y_hat, y)
                loss += self.feature_loss(fmap_r, fmap_g)
                loss += self.generator_loss(g_out)
                print(f"Generator loss: {loss}")
                return loss
        

    def feature_loss(self, fmap_r, fmap_g):
        loss = 0
        for dr, dg in zip(fmap_r, fmap_g):
            for rl, gl in zip(dr, dg):
                loss += torch.mean(torch.abs(rl - gl))
        return loss


    def discriminator_loss(self, disc_real_outputs, disc_generated_outputs):
        loss = 0
        for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
            r_loss = torch.clamp(1-dr, 0, float('inf'))
            r_loss = torch.mean(r_loss)
            g_loss = torch.clamp(1+dg, 0, float('inf'))
            g_loss = torch.mean(g_loss)
            loss += (r_loss + g_loss)
        return loss


    def generator_loss(self, disc_outputs):
        loss = 0
        for dg in disc_outputs:
            l = torch.mean(-dg)
            loss += l
        return loss
    

    def stft_loss(self, pred, real, nfft = 400, nhop = 200, nwin = 200, window="hann_window", phase=False):
        if pred.dim() > 2:
            pred = pred.squeeze(1)
        if real.dim() > 2:
            real = real.squeeze(1)
        
        window = getattr(torch, window)(nwin).to(pred.device)
        pred_stft = torch.stft(pred, nfft, nhop, nwin, window)
        real_stft = torch.stft(real, nfft, nhop, nwin, window)
        pred_mag = torch.sqrt(torch.clamp(pred_stft[...,0] ** 2 + pred_stft[...,1] ** 2, min=1e-14)).unsqueeze(-1)
        real_mag = torch.sqrt(torch.clamp(real_stft[...,0] ** 2 + real_stft[...,1] ** 2, min=1e-14)).unsqueeze(-1)
        if phase:
            pred_phase = pred_stft / pred_mag
            real_phase = real_stft / real_mag
            pred_mag = pred_mag**(0.3)
            real_mag = real_mag**(0.3)
            logmag_loss = 0.7*torch.mean(torch.abs(pred_mag - real_mag)) + 0.3*torch.mean(torch.abs(pred_mag*pred_phase - real_mag*real_phase))
        else:
            logmag_loss = torch.mean(torch.abs(torch.log(pred_mag) - torch.log(real_mag)))
        spectralconverage_loss = torch.mean(torch.norm(pred_mag - real_mag, p="fro") / torch.norm(pred_mag, p="fro"))
        loss = logmag_loss + spectralconverage_loss

        # if pred.dim() > 2:
        #     pred = pred.squeeze(1)
        # if real.dim() > 2:
        #     real = real.squeeze(1)
        # loss = -cal_si_snr(pred, real)
        return loss


def receptive_field_size(kernel_size = 3, num_layers = 10, num_stacks = 2, sample_rate = 16000):
    s = [2**i * (kernel_size-1) for i in range(num_layers)]
    return (sum(s) * num_stacks * 15 + 1) / sample_rate

if __name__ == '__main__':
    print(f"receptive field size: {receptive_field_size()} s")
    import time

    with torch.no_grad():
        # model = Hifi_GAN(input_channel=3, channel=128, sample_rate=16000, nffts=[400, 800, 1600], n_mels=80, 
        #                 num_layers=10, num_stacks = 2, kernel_size = 3, post_channel = 128, post_layers = 12)
        model = Hifi_GAN(nffts=[400,800,1600], n_mels=80, num_channels=[16, 32, 64, 128], num_freqs=201, hidden=512, 
                            segment_length = 3200, num_layers = 2, num_inputs=3, kernel_size=3, dropout=0.0)


        x = torch.rand(1, 3, 27723)  # 1.6s
        y = torch.rand(1, 1, 27723)
        x_len = x.shape[-1]
        
        #Stage1
        start = time.time()
        loss = model.train_stage(x, y, stage=1)
        print(f"Real Time Process for {x.shape[-1]/16000}s: {time.time() - start}")
        print(loss)

        #Stage2
        start = time.time()
        loss = model.train_stage(x, y, stage=2)
        print(f"Real Time Process for {x.shape[-1]/16000}s: {time.time() - start}")
        print(loss)

        #Stage3
        start = time.time()
        loss_d, y_hat, y_before = model.train_stage(x, y, stage=3)
        loss_g = model.train_stage(x, y, stage=3, y_hat = y_hat, y_before = y_before)
        print(f"Real Time Process for {x.shape[-1]/16000}s: {time.time() - start}")
        print(loss_d, loss_g)
        
        parameters = sum(param.numel() for param in model.generator.parameters())
        print(str(parameters / 10**6) +' MB ')
        parameters = sum(param.numel() for param in model.discriminator.parameters())
        print(str(parameters / 10**6) +' MB ')
