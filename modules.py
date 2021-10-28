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
        if time:
            with torch.no_grad():
                self.step += T
                self.mean = global_mean.clone()
                self.var = global_var.clone()
    
    def reset(self):
        if self.time:
            self.mean.zero_()
            self.var.zero_()
            self.step.zero_()


class CumLayerNorm(nn.Module):
    def __init__(self, dims):
        super(CumLayerNorm, self).__init__()
        self.register_buffer('step', torch.tensor(0))
        self.register_buffer('mean', torch.tensor(0))


    def forward(self, x):
        # x = B x C x F x F
        if x.dim() == 4:
            mean = torch.mean(x, (1,2,3), keepdim=True)
        else:
            mean = torch.mean(x, (1,2), keepdim=True)
        
        if self.mean is None:
            self.mean = mean
        else:
            alpha = self.step / (self.step + 1)
            mean = alpha * self.mean + (1.0 - alpha) * mean
        self.step += 1
        x /= mean + EPS
        with torch.no_grad():
            self.mean = mean
        return x
    
    def reset(self):
        self.mean.zero_()
        self.step.zero_()

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
        
        self.ql = nn.Linear(model_dim, model_dim)
        self.kl = nn.Linear(model_dim, model_dim)
        self.vl = nn.Linear(model_dim, model_dim)
        self.linear = nn.Linear(model_dim, model_dim)
        self.softmax = nn.Softmax(dim=-1)
        
        self.delta = nn.Parameter(torch.ones(1))
        ind = torch.arange(1, maxlen+1).unsqueeze(1).repeat(1,maxlen)    
        self.G = torch.exp(-(ind-ind.transpose(0,1))**2 / (self.delta**2 + EPS)).unsqueeze(0)
        self.d = torch.sqrt(torch.tensor(self.model_dim))

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
        #[E*B, T, F/E] * [E*B, F/E, maxlen]
        x = torch.matmul(q, k.transpose(1,2)) * self.G[:, -T:] / self.d
        #[E*B, T, maxlen] * [E*B, maxlen, F/E]
        x = torch.matmul(self.softmax(x), v)

        x = torch.cat(torch.split(x, B, dim=0), dim=-1)
        x = self.linear(x)

        with torch.no_grad():
            self.bk = k.clone()
            self.bv = v.clone()
        return x
    
    def reset(self):
        self.bk.zero_()
        self.bv.zero_()


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
        self.norm_a =  GlobalLayerNorm(model_dim, last=True)
        self.activation = nn.ReLU()
        self.linear_in = nn.Linear(model_dim, fn_dim)
        self.linear_out = nn.Linear(fn_dim, model_dim)
        self.norm_i = GlobalLayerNorm(model_dim, last=True)
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


class SequenceModel(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, num_layers, bidirectional, linear=True,
                    sequence_model="GRU",output_activate_function="Tanh"):
    
        super().__init__()
        """
        ---------------------------------------
        import torch
        from modules import SequenceModel
        model = SequenceModel(20, 10, 10, 2, bidirectional=False)
        x = torch.rand(3, 20, 6)
        y = model(x)
        print(x.shape, model.h)
        """
        
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
        if bidirectional:
            self.fc_output_layer = nn.Linear(hidden_size * 2, output_size)
        else:
            self.fc_output_layer = nn.Linear(hidden_size, output_size)

        self.norm = CumLayerNorm(output_size)
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
        self.register_buffer('h', None)

    
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
        o = o.permute(0, 2, 1).contiguous()  # [B, T, F] => [B, F, T]
        if self.output_activate_function:
            o = self.activate_function(o)
        o = self.norm(o)

        with torch.no_grad():
            self.h = h
        return o

    def reset(self):
        self.h = None
        self.norm.reset()


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
        self.norm = GlobalLayerNorm(n_outputs)
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
        self.residualnorm = CumLayerNorm(n_outputs)
        self.sigmoid = nn.Sigmoid()
        self.residual = nn.Conv2d(n_outputs, n_outputs, (1,1))
        self.relu = nn.ReLU()
        self.norm = CumLayerNorm(n_outputs)

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


class Complex_GlobalLayerNorm(nn.Module):
    '''
    import torch
    from modules import Complex_GlobalLayerNorm
    norm = Complex_GlobalLayerNorm(10)
    x = torch.rand(3, 10, 201, 6, 2)
    y = norm(x)
    print(x.shape, norm.step)
    '''
    def __init__(self, dim, last = False, time = True):
        super(Complex_GlobalLayerNorm, self).__init__()
        self.time = time
        if time:
            self.register_buffer('step', torch.tensor(0))
            self.register_buffer('mean', torch.zeros(1, 1, 1, 1, 2))
            self.register_buffer('var_rr', torch.tensor(0))
            self.register_buffer('var_ii', torch.tensor(0))
            self.register_buffer('var_ri', torch.tensor(0))
        
        if last:
            self.weight = nn.Parameter(torch.ones(1, 1, 1, dim, 2))
            self.bias = nn.Parameter(torch.zeros(1, 1, 1, dim, 2))
        else:
            self.weight = nn.Parameter(torch.ones(1, dim, 1, 1, 2))
            self.bias = nn.Parameter(torch.zeros(1, dim, 1, 1, 2))


    def forward(self, x):
        # x = B x C x F x T x 2
        T = x.shape[3]
        mean = torch.mean(x, (1,2,3), keepdim=True)
        xp = x - mean
        var_rr = torch.mean(xp[...,0] ** 2, (1,2,3), keepdim=True)
        var_ii = torch.mean(xp[...,1] ** 2, (1,2,3), keepdim=True)
        var_ri = torch.mean(xp[...,0] * xp[...,1], (1,2,3), keepdim=True)
        

        if self.time:
            alpha = self.step / (self.step + T)
            global_mean = alpha * self.mean + (1.0 - alpha) * mean
            history_delta = global_mean - self.mean
            now_delta = global_mean - mean
            global_var_rr = alpha * (self.var_rr + (global_mean[...,0] - self.mean[...,0]) ** 2) + \
                            (1.0 - alpha) * (var_rr + (global_mean[...,0] - mean[...,0]) ** 2)
            global_var_ii = alpha * (self.var_ii + (global_mean[...,1] - self.mean[...,1]) ** 2) + \
                            (1.0 - alpha) * (var_ii + (global_mean[...,1] - mean[[...,1]]) ** 2)
            global_var_ri = alpha * (self.var_ri + (global_mean[...,0] - self.mean[...,0]) * (global_mean[...,1] - self.mean[...,1])) + \
                            (1.0 - alpha) * (var_ri + (global_mean[...,0] - mean[...,0]) * (global_mean[...,1] - mean[...,1]))
        else:
            global_mean = mean
            global_var_rr = var_rr
            global_var_ii = var_ii
            global_var_ri = var_ri


        det = global_var_rr*global_var_ii-global_var_ri.pow(2)
        s = torch.sqrt(det + EPS)
        t = torch.sqrt(global_var_rr + global_var_ii + 2*global_var_ri + EPS)
        inverse_st = 1.0 / (s * t + EPS)
        Rrr = (global_var_ii + s) * inverse_st
        Rii = (global_var_rr + s) * inverse_st
        Rri = -global_var_ri * inverse_st

        x[...,0] = Rrr * xp[...,0] + Rri * xp[...,1]
        x[...,1] = Rri * xp[...,0] + Rii * xp[...,1]
        x = x * self.weight + self.bias
        
        with torch.no_grad():
            self.step += T
            self.mean = global_mean.clone()
            self.var_rr = global_var_rr.clone()
            self.var_ii = global_var_ii.clone()
            self.var_ri = global_var_ri.clone()
        return x
    
    def reset(self):
        if self.time:
            self.step.zero_()
            self.mean.zero_()
            self.var_rr.zero_()
            self.var_ii.zero_()
            self.var_ri.zero_()


class Complex_SequenceModel(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, num_layers, bidirectional,
                 sequence_model="GRU",output_activate_function="Tanh"):
        super().__init__()
        """
        ---------------------------------------
        import torch
        from modules import Complex_SequenceModel
        model = Complex_SequenceModel(20, 20, 10, 2, bidirectional=False)
        x = torch.rand(3, 20, 6, 2)
        y = model(x)
        print(x.shape, model.h)
        """
        self.model_type = sequence_model

        # Sequence layer
        if sequence_model == "LSTM":
            self.real_seq = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, 
                                batch_first=True, bidirectional=bidirectional)   
            self.img_seq = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, 
                                batch_first=True, bidirectional=bidirectional) 
        elif sequence_model == "GRU":
            self.real_seq = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
                                batch_first=True, bidirectional=bidirectional)
            self.img_seq = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
                                batch_first=True, bidirectional=bidirectional)

        # Fully connected layer
        if bidirectional:
            self.real_layer = nn.Linear(hidden_size * 2, output_size)
            self.img_layer = nn.Linear(hidden_size * 2, output_size)
        else:
            self.real_layer = nn.Linear(hidden_size, output_size)
            self.img_layer = nn.Linear(hidden_size, output_size)

        self.norm = Complex_GlobalLayerNorm(output_size, last=True)
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
        self.register_buffer('h_rr', None)
        self.register_buffer('h_ii', None)
        self.register_buffer('h_ri', None)
        self.register_buffer('h_ir', None)

    def forward(self, x):
        """
        Args:
            x: [B, F, T, 2]
        Returns:
            [B, F, T, 2]
        """
        assert x.dim() == 4
        self.real_seq.flatten_parameters()
        self.img_seq.flatten_parameters()

        # contiguous 使元素在内存中连续，有利于模型优化，但分配了新的空间
        # 建议在网络开始大量计算前使用一下
        x = x.transpose(1, 2).contiguous()  # [B, F, T, 2] => [B, T, F, 2]
        rr, h_rr = self.real_seq(x[...,0], self.h_rr)
        ii, h_ii = self.img_seq(x[...,1], self.h_ii)
        ri, h_ri = self.real_seq(x[...,1], self.h_ri)
        ir, h_ir = self.img_seq(x[...,0], self.h_ir)
        real = rr - ii
        img = ri + ir

        x0 = self.real_layer(real) - self.img_layer(img)
        x1 = self.real_layer(img) + self.img_layer(real)
        x = torch.stack([x0, x1], dim=-1)
        
        if self.output_activate_function:
            x = self.activate_function(x)
        x = self.norm(x.unsqueeze(1)).squeeze(1)
        x = x.transpose(1, 2).contiguous()  # [B, T, F, 2] => [B, F, T, 2]
        
        with torch.no_grad():
            self.h_rr = h_rr
            self.h_ii = h_ii
            self.h_ri = h_ri
            self.h_ir = h_ir
        return x

    def reset(self):
        self.h_rr = None
        self.h_ii = None
        self.h_ri = None
        self.h_ir = None


class Complex_TemporalConv2d(nn.Module):
    """
    ---------------------------------------
    import torch
    from modules import Complex_TemporalConv2d
    model = Complex_TemporalConv2d(1, 16, 3, 1, (1,2), (1,4))
    x = torch.rand(3, 1, 20, 10, 2)
    y = model(x)
    print(x.shape, model.real_buffer[0, 0, 0, :])
    """
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.0, activation="ReLU"):
        super(Complex_TemporalConv2d, self).__init__()
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
        self.real_conv = nn.Conv2d(n_inputs, n_outputs, kernel_size, stride=stride, padding=(padding[0], 0), dilation=dilation)
        self.img_conv = nn.Conv2d(n_inputs, n_outputs, kernel_size, stride=stride, padding=(padding[0], 0), dilation=dilation)
        self.dropout = nn.Dropout(dropout)
        self.norm = Complex_GlobalLayerNorm(n_outputs)
        self.register_buffer('real_buffer', None)
        self.register_buffer('img_buffer', None)

    def forward(self, x):
        #x: [B, C, F, T, 2]
        #return: [B, C', F, T, 2]
        B, C, F, T, _ = x.shape
        if self.real_buffer is None:
            with torch.no_grad():
                self.real_buffer = torch.zeros((B, C, F, self.padding), device=x.device, dtype=x.dtype)
                self.img_buffer = torch.zeros((B, C, F, self.padding), device=x.device, dtype=x.dtype)

        real = torch.cat([self.real_buffer, x[...,0]], dim=-1)
        img = torch.cat([self.img_buffer, x[...,1]], dim=-1)
        oreal = self.real_conv(real) - self.img_conv(img)
        oimg = self.real_conv(img) + self.img_conv(real)
        out = torch.stack([oreal, oimg], dim=-1)
        out = self.dropout(out)
        out = self.activation(out) 
        out = self.norm(out)
        with torch.no_grad():
            if T > self.padding:
                self.real_buffer = real[..., -self.padding:]
                self.img_buffer = img[..., -self.padding:]
            else:
                self.real_buffer[..., :self.padding - T] = self.real_buffer[..., T-self.padding:]
                self.real_buffer[..., -T:] = real
                self.img_buffer[..., :self.padding - T] = self.img_buffer[..., T-self.padding:]
                self.img_buffer[..., -T:] = img
        return out
    
    def reset(self):
        self.real_buffer = None
        self.img_buffer = None
        self.norm.reset()


class Complex_TemporalConvTranspose2d(nn.Module):
    """
    ---------------------------------------
    import torch
    from modules import Complex_TemporalConvTranspose2d
    model = Complex_TemporalConvTranspose2d(16, 1, 3, 1, (1,2), (1,4))
    x = torch.rand(3, 16, 20, 10, 2)
    res = torch.rand(3, 1, 20, 10, 2)
    y = model(x, res)
    """
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.0, activation="ReLU"):
        super(Complex_TemporalConvTranspose2d, self).__init__()
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
        self.real_deconv = nn.ConvTranspose2d(n_inputs, n_outputs, kernel_size, stride=stride, padding=(padding[0], 0), dilation=dilation)
        self.img_deconv = nn.ConvTranspose2d(n_inputs, n_outputs, kernel_size, stride=stride, padding=(padding[0], 0), dilation=dilation)
        self.dropout = nn.Dropout(dropout)

        self.real_residualmask = nn.Conv2d(n_outputs, n_outputs, (1,1))
        self.img_residualmask = nn.Conv2d(n_outputs, n_outputs, (1,1))
        self.residualnorm = Complex_GlobalLayerNorm(n_outputs)
        self.sigmoid = nn.Sigmoid()

        self.real_residual = nn.Conv2d(n_outputs, n_outputs, (1,1))
        self.img_residual = nn.Conv2d(n_outputs, n_outputs, (1,1))
        self.norm = Complex_GlobalLayerNorm(n_outputs)
        self.relu = nn.ReLU()
        

    def forward(self, x, res=None):
        #x: [B, C, F, T, 2]
        #return: [B, C', F, T, 2]
        T = x.shape[-2]
        real = (self.real_deconv(x[...,0]) - self.img_deconv(x[...,1]))[..., -T:]
        img = (self.real_deconv(x[...,1]) + self.img_deconv(x[...,0]))[..., -T:]
        out = torch.stack([real, img], dim=-1)
        out = self.activation(out)
        out = self.norm(out)

        if res is not None:
            B, C, F, T, _ = res.shape
            if F > out.shape[2]:
                pad = torch.zeros((B, C, F-out.shape[2], T, 2), device=out.device, dtype=out.dtype)
                out = torch.cat([out, pad], dim=2)
            elif F < out.shape[2]:
                out = out[:, :, :F]
            
            real = self.real_residualmask(res[...,0]) - self.img_residualmask(res[...,1])
            img = self.real_residualmask(res[...,1]) + self.img_residualmask(res[...,0])
            mask = torch.stack([real,img], dim=-1)
            mask = self.sigmoid(mask)

            real = self.real_residual(res[...,0]) - self.img_residual(res[...,1])
            img = self.real_residual(res[...,1]) + self.img_residual(res[...,0])
            res = torch.stack([real, img], dim=-1)
            out = mask * self.relu(res) + (1.0-mask) * out
            out = self.residualnorm(out)
        return out
    
    def reset(self):
        self.norm.reset()
        self.residualnorm.reset()

