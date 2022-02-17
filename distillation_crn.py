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

        x = (x - global_mean) / (torch.sqrt(global_var) + EPS) * self.weight + self.bias
        
        self.step += T
        self.mean = global_mean.detach()
        self.var = global_var.detach()
        return x
    
    def reset(self):
        self.mean = None
        self.var = None
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
            elif output_activate_function == "ELU":
                self.activate_function = nn.ELU()
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
        #self.sequence_model.flatten_parameters()

        # contiguous 使元素在内存中连续，有利于模型优化，但分配了新的空间
        # 建议在网络开始大量计算前使用一下
        x = x.permute(0, 2, 1).contiguous()  # [B, F, T] => [B, T, F]
        o, h = self.sequence_model(x, self.h)
        if self.linear:
            o = self.fc_output_layer(o)
        feature = o
        if self.output_activate_function:
            o = self.activate_function(o)
        o = self.norm(o.unsqueeze(1)).squeeze(1)
        o = o.permute(0, 2, 1).contiguous()  # [B, T, F] => [B, F, T]

        
        if self.model_type == "LSTM":
            self.h = (h[0].detach(), h[1].detach())
        else:
            self.h = h.detach()
        return o, feature

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
        elif activation == "ELU":
            self.activation = nn.ELU()
        else:
            raise NotImplementedError(f"Not implemented activation function {activation}")
        
        #手动padding
        self.conv = nn.Conv2d(n_inputs, n_outputs, kernel_size, stride=stride, padding=(padding[0], 0), dilation=dilation)
        self.conv_trans = nn.Conv2d(n_outputs, n_outputs, 1, stride=1, padding=0)
        self.conv_gated = nn.Conv2d(n_outputs, n_outputs, 1, stride=1, padding=0)
        self.dropout = nn.Dropout(dropout)
        self.sigmoid = nn.Sigmoid()
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
        feature = self.net(inp)
        out = self.activation(feature)
        out = self.conv_trans(out) * self.sigmoid(self.conv_gated(out))
        out = self.norm(out)
        if T > self.padding:
            self.buffer = x[..., -self.padding:].clone().detach()
        else:
            self.buffer[..., :self.padding - T] = self.buffer[..., T-self.padding:]
            self.buffer[..., -T:] = x.clone().detach()
        return out, feature
    
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
        elif activation == 'ELU':
            self.activation = nn.ELU()
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
        feature = out
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
            out = mask * self.activation(self.residual(res)) + (1.0-mask) * out
        return out, feature
    
    def reset(self):
        self.norm.reset()
        self.residualnorm.reset()


class TemporalCRN(nn.Module):
    """
    ---------------------------------------
    Use ELU
    Add Frequency Dilation Convolution (preconv)
    Add TemporalCNN Gated Conv
    """
    def __init__(self, num_channels, num_freqs, hidden, segment_length,
                    num_layers = 1, num_inputs=3, kernel_size=3, dropout=0.0, sample_rate=16000, 
                    win_length=25, hop_length=10, n_fft=400):
        super(TemporalCRN, self).__init__()
        self.segment_length = segment_length
        self.num_freqs = num_freqs
        self.num_channels = num_channels
        activation = "ELU"

        self.stft = STFT(sample_rate=sample_rate, win_length=win_length, 
                hop_length=hop_length, n_fft=n_fft)

        self.istft = ISTFT(sample_rate=sample_rate, win_length=win_length, 
                        hop_length=hop_length, n_fft=n_fft)

        preconvs = []
        frequency_dilations = [1, 2, 4, 8]
        for i in range(3):
            preconvs += [TemporalConv2d((2 * num_inputs - 1), (2 * num_inputs - 1), (5,5), stride=(1,1), dilation=(frequency_dilations[i],1),
                            padding=(2 * frequency_dilations[i], 4), dropout=dropout, activation=activation)]
        self.preconvlist = nn.ModuleList(preconvs)

        convs = []
        deconvs = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = (2 * num_inputs - 1) if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            convs += [TemporalConv2d(in_channels, out_channels, (5,kernel_size), stride=(2,1), dilation=(1,dilation_size),
                                     padding=(2, (kernel_size-1) * dilation_size), dropout=dropout, activation=activation)]
            
            dilation_size = 2 ** (num_levels-i-1)
            if i==0:
                deconvs = [TemporalConvTranspose2d(out_channels, 2, (5,kernel_size), stride=(2,1), dilation=(1,dilation_size),
                            padding=(2, (kernel_size-1) * dilation_size), dropout=dropout, activation=activation)]
            else:
                deconvs = [TemporalConvTranspose2d(out_channels, in_channels, (5,kernel_size), stride=(2,1), dilation=(1,dilation_size),
                            padding=(2, (kernel_size-1) * dilation_size), dropout=dropout, activation=activation)] + deconvs
        
        
        self.convlist = nn.ModuleList(convs)
        self.deconvlist = nn.ModuleList(deconvs)

        self.gru = SequenceModel((num_freqs // 2**num_levels + 1)*num_channels[-1], (num_freqs // 2**num_levels + 1)*num_channels[-1], hidden, num_layers, False, 
                                 linear=True, sequence_model="GRU",output_activate_function=activation)

    def forward(self, x):
        #[B, M, F, T, 2]
        noisy = x[:,0]
        angle = torch.arctan(x[..., 1]/(x[..., 0] + EPS) + EPS)
        angle = angle[:, 0].unsqueeze(1) - angle[:, 1:]
        mag = torch.sqrt(x[..., 0]**2 + x[..., 1]**2 + 1e-10)
        x = torch.cat([mag, angle], dim = 1)
        feature = []
        for m in self.preconvlist:
            res = x
            x, _ = m(x)
            x += res

        #[B, 2M-1, F, T]
        residuals = [x]
        index = 0
        for m in self.convlist:                
            x, ft = m(x)
            residuals += [x]
            index += 1

        feature += [ft]
        B, C, F, T = x.shape
        #[B, C, F, T] -> [B, C*F, T]
        x = x.reshape(B, C*F, T).contiguous()
        x, ft = self.gru(x)
        #[B, C, F, T]
        x = x.reshape(B, C, F, T)
        feature += [ft.reshape(B, C, F, T)]

        B, C, F, T = x.shape
        index = -2
        for m in self.deconvlist[:-1]:
            x, ft = m(x, residuals[index])
            feature += [ft]
            index -= 1
        #last one, [B, 2, F, T] -> [B, F, T, 2]
        x, _ = self.deconvlist[-1](x)
        x = x.permute(0, 2, 3, 1)
        
        x = decompress_cIRM(x)
        enhanced_real = x[...,0] * noisy[...,0] - x[...,1] * noisy[...,1]
        enhanced_imag = x[...,1] * noisy[...,0] + x[...,0] * noisy[...,1]
        #[B, F, T, 2]
        x = torch.stack([enhanced_real, enhanced_imag], dim=-1)
        return x, feature
    
    def get_channel_num(self):
        return [self.num_channels[-1], self.num_channels[-1], self.num_channels[2], self.num_channels[1], self.num_channels[0]]
    
    def reset(self):
        for m in self.preconvlist:
            m.reset()
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
        B, C, T = mixture.shape
        if not flag:
            pad = torch.zeros((B, C, self.segment_length // 2), dtype=mixture.dtype, device=mixture.device)
            mixture = torch.cat([pad, mixture], dim=-1)
        x, gap = self.preprocessing(mixture)
        #N, B, C, F, T, 2
        N, B, C, F, T, _ = x.shape
        if not flag:
            self.reset()
        ft = [[] for _ in range(len(self.get_channel_num()))]
        pred_source = torch.zeros((N, B, F, T, 2), dtype=x.dtype, device=x.device)
        #[N, B, C, F, T]
        for idx in range(N):
            inp = x[idx]
            #with autocast():
            preds, feature = self.forward(inp)
            pred_source[idx] = preds
            for i in range(len(feature)):
                ft[i].append(feature[i])
        #[K], N*B, Ck, Fk, T
        ft = [torch.cat(f,dim=0) for f in ft]
        #[N, B, F, T, 2] -> [B, L]
        pred_source = self.postprocessing(pred_source, gap)
        if not flag:
            pred_source = pred_source[..., self.segment_length // 2: ]
        return pred_source, ft
    
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

        #实际上是stoi
        stoi = stoi_loss(source, pred_source, length)
        sisnr = -cal_si_snr(pred_source, source, length) 
        stoi = stoi.to(sisnr.device)
        loss = 0.7 * stoi + 0.3 * sisnr
        print(sisnr)
        if torch.isnan(loss):
            stoi = stoi.fill_(0.0)
            sisnr = sisnr.fill_(0.0)
            loss = loss.fill_(0.0)
        return loss, stoi, sisnr


class DistillationCRN(nn.Module):
    """
    ---------------------------------------
    Use ELU
    Add Frequency Dilation Convolution (preconv)
    Add TemporalCNN Gated Conv
    """
    def __init__(self, *args, **kargs):
        super(DistillationCRN, self).__init__()
        model_path = None
        if 'path' in kargs.keys():
            model_path = kargs['path']
            del kargs['path']
        self.teacher = TemporalCRN(*args, **kargs)
        if model_path is not None:
            self.teacher.load_state_dict(torch.load(model_path))
            self.teacher.eval()
            for param in self.teacher.parameters():
                param.requires_grad = False

        kargs['num_channels'] = [16, 32, 64, 64]
        kargs['hidden'] = 128
        self.student = TemporalCRN(*args, **kargs)
        for pt, ps in zip(self.teacher.parameters(), self.student.parameters()):
            if ps.shape == pt.shape:
                ps.data = pt.data

        t_channels = self.teacher.get_channel_num()
        s_channels = self.student.get_channel_num()
        self.connectors = nn.ModuleList([self.build_feature_connector(t, s) for t, s in zip(t_channels, s_channels)])

    def build_feature_connector(self, t_channel, s_channel):
        C = [nn.Conv2d(s_channel, t_channel, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(t_channel)]

        for m in C:
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        return nn.Sequential(*C)

    def get_margin(self, ft):
        mask = (ft < 0.0).float()
        masked_ft = ft * mask
        margin = masked_ft.sum(dim=(0,2,3), keepdim=True) / (mask.sum(dim=(0,2,3), keepdim=True)+EPS)
        return margin

    def distillation_loss(self, ft, fs):
        loss = 0.0
        for i in range(len(ft)):
            t,s = ft[i], fs[i]
            trans = self.connectors[i]
            margin = self.get_margin(t)
            t = torch.max(t, margin)
            s = trans(s)
            mask = 1.0 - ((s <= t) & (t <= 0.0)).float()
            loss += torch.mean((s - t)**2 * mask)
        return loss / len(ft)

    def forward(self, noisy, clean, length, flag):
        _, ft = self.teacher.realtime_process(noisy, flag)
        pred, fs = self.student.realtime_process(noisy, flag)
        loss, stoi, sisnr = self.student.compute_loss(clean, pred, length)
        loss += self.distillation_loss(ft, fs)
        return loss, stoi, sisnr


if __name__ == "__main__":
    import time

    with torch.no_grad():
        # model = TemporalCRN(num_channels=[16, 32, 64, 128], num_freqs=201, hidden=512, 
        #                     segment_length = 3200, num_layers = 2, num_inputs=3, kernel_size=3, dropout=0.0)
        model = DistillationCRN(num_channels=[16, 32, 64, 128], num_freqs=201, hidden=512, 
                                segment_length = 3200, num_layers = 2, num_inputs=3, kernel_size=3, dropout=0.0, path = './modules/denoise/crn_elu/TemporalCRN.pth').student
        ipt = torch.rand(1, 3, 160000)  # 1.6s
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
        
        pred_source, feature = model.realtime_process(ipt)
        print(f"Real Time Process for {ipt.shape[-1]/16000}s: {time.time() - start}")
        print(pred_source.shape)
        print(feature[0].shape, feature[1].shape, feature[2].shape)
        
        parameters = sum(param.numel() for param in model.parameters())
        print(str(parameters / 10**6) +' MB ')
    
    # distillation = DistillationCRN(num_channels=[16, 32, 64, 128], num_freqs=201, hidden=512, 
    #                                 segment_length = 3200, num_layers = 2, num_inputs=3, kernel_size=3, dropout=0.0, path = './modules/denoise/crn_elu/TemporalCRN.pth')
    # loss, stoi, sisnr = distillation(ipt, ipt[:,0], torch.tensor([ipt_len]))
    # print(loss, stoi, sisnr)
