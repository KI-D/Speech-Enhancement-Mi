import torch
from torch.nn import functional
from torch import nn
from torch_complex import ComplexTensor
import torch_complex
from torch.cuda.amp import autocast as autocast
import torch.nn.init as init
from utility import *
from speechbrain.processing.features import STFT,ISTFT


EPS = 1e-8
def trace(
        complex_matrix: ComplexTensor
) -> ComplexTensor:
    """
    Return trace of a complex matrices
    """
    mat_size = complex_matrix.size()
    diag_index = torch.eye(mat_size[-1], dtype=torch.bool, device=complex_matrix.device).expand(*mat_size)
    complex_trace = ComplexTensor(complex_matrix.real.masked_select(diag_index).view(*mat_size[:-1]).sum(-1).contiguous(),
                    complex_matrix.imag.masked_select(diag_index).view(*mat_size[:-1]).sum(-1).contiguous())
    return complex_trace

class MVDRBeamformer(nn.Module):
    """
    MVDR (Minimum Variance Distortionless Response) beamformer
    """

    def __init__(self, use_mask_norm: bool = False, eps: float = EPS):
        super().__init__()
        self.use_mask_norm = use_mask_norm
        self.eps = eps

    def apply_beamformer(self, beamforming_vector: ComplexTensor, mixture: ComplexTensor) -> ComplexTensor:
        """Apply beamforming weights at frame level

        Args:
            beamforming_vector: beamforming weighted vector with shape of [..., C]
            mixture: mixture of shape [..., C, F ,T] -> [...,C, T]

        Notes:
            There's no relationship between frequencies.

        Returns:
            [..., T]
        """
        # [B, F, C, T]
        mixture = mixture.permute(0,2,1,3)
        # [..., C] x [..., C, T] => [..., T]
        es = torch_complex.functional.einsum("bfc, bfct -> bft", [beamforming_vector.conj(), mixture])
        return es

    @staticmethod
    def stabilize_complex_number(complex_matrix: ComplexTensor, eps: float = EPS):
        return ComplexTensor(complex_matrix.real, complex_matrix.imag + torch.tensor(eps))

    def _derive_weight(self, speech_psd: ComplexTensor, noise_psd: ComplexTensor, reference_vector: torch.Tensor,
                       eps: float = 1e-8) -> ComplexTensor:
        """
        Derive MVDR beamformer

        Args:
            speech_psd: [B, F, C, C]
            noise_psd: [B, F, C, C]
            reference_vector: [B x C], reference selection vector
            eps:

        Return:
            [B, C, F]

        Examples:
        import torch
        from torch.nn import functional
        from torch import nn
        from torch_complex import ComplexTensor
        import torch_complex
        from fullsubnet import MVDRBeamformer
        B = 2
        C = 8
        F = 257
        T = 200
        sm = ComplexTensor(torch.rand(B, C, F, T), torch.rand(B, C, F, T))
        nm = ComplexTensor(torch.rand(B, C, F, T), torch.rand(B, C, F, T))
        c = ComplexTensor(torch.rand(B, C, F, T), torch.rand(B, C, F, T))
        mvdr_beamformer = MVDRBeamformer()
        output = mvdr_beamformer(sm, nm, c)
        print(output.shape)
        """
        _, _, _, num_channels = noise_psd.shape

        identity_matrix = torch.eye(num_channels, device=noise_psd.device, dtype=noise_psd.dtype)
        noise_psd = noise_psd + identity_matrix * eps
        # [B, F, C, C]
        noise_psd_inverse = self.stabilize_complex_number(noise_psd.inverse())
        # [B, F, C, C]
        # einsum("...ij,...jk->...ik", Rn_inv, Rs)
        noise_psd_inverse_speech_psd = noise_psd_inverse @ speech_psd
        # [B, F]
        trace_noise_psd_inverse_speech_psd = trace(noise_psd_inverse_speech_psd) + eps
        # [B, F, C]
        # einsum("...fnc,...c->...fn", Rn_inv_Rs, u)
        noise_psd_inverse_speech_psd_u = (noise_psd_inverse_speech_psd @ reference_vector[:, None, :, None]).sum(-1)
        # [B, F, C]
        weight = noise_psd_inverse_speech_psd_u / trace_noise_psd_inverse_speech_psd[..., None]
        # [B, F, C]
        return weight

    @staticmethod
    def mask_norm(mask: ComplexTensor) -> ComplexTensor:
        #[B, C, F, T]
        norms = torch.sqrt(mask.real**2 + mask.imag**2 + EPS)
        max_abs = torch.max(norms, dim=2, keepdims=True)[0]
        mask = mask / (max_abs + EPS)
        return mask

    @staticmethod
    def estimate_psd(mask: ComplexTensor, complex_matrix: ComplexTensor, eps: float = 1e-5) -> ComplexTensor:
        """
        Power Spectral Density Covariance (PSD) estimation

        Args:
            mask: [B, C, F, T], CTF-masks
            complex_matrix: [B, C, F, T], complex-valued matrix of short-term Fourier transform coefficients
            eps:

        Return:
            [B, F, C, C]
        """
        # [B, C, F, T] => [B, F, C, T]
        complex_matrix = complex_matrix.transpose(1, 2)
        # [B, F, T] => [B, F, 1, T]
        mask = mask.unsqueeze(2)
        signal = complex_matrix * mask
        # [B, F, C, C]: einsum("...it,...jt->...ij", spec * mask, spec.conj())
        nominator = signal @ signal.conj().transpose(-1, -2)
        # [B, F, C, T] => [B, F, C, C]
        denominator = mask.conj() @ mask.transpose(-1, -2)
        # [B, F, C, C]
        psd = nominator / (denominator+EPS)
        # stabilize
        return ComplexTensor(psd.real, psd.imag + torch.tensor(eps))

    def forward(self, speech_mask: ComplexTensor, noise_mask: ComplexTensor, complex_matrix: ComplexTensor) -> ComplexTensor:
        """
        Args:
            speech_mask: [B, C, F, T], real-valued speech T-F mask
            noise_mask: [B, C, F, T], real-valued noise T-F mask
            complex_matrix: [B x C x F x T], noisy complex spectrogram

        Return:
            [B, F, T], enhanced complex spectrogram
        """
        batch_size, num_channels, _, _ = complex_matrix.shape

        # B x F x T
        if self.use_mask_norm:
            speech_mask = self.mask_norm(speech_mask)
            noise_mask = self.mask_norm(noise_mask)

        # [B, F, C, C]
        speech_psd = self.estimate_psd(speech_mask, complex_matrix)
        noise_psd = self.estimate_psd(noise_mask, complex_matrix)

        # [B, C]
        reference_vector = torch.zeros((batch_size, num_channels), device=noise_psd.device, dtype=noise_psd.dtype)
        reference_vector[:, 0].fill_(1)

        # [B, C, F]
        weight = self._derive_weight(speech_psd, noise_psd, reference_vector, eps=self.eps)

        # [B, F, T]
        filtered_complex_matrix = self.apply_beamformer(weight, complex_matrix)
        return filtered_complex_matrix


class CumLayerNorm(nn.Module):
    def __init__(self):
        super(CumLayerNorm, self).__init__()
        self.mean = None
        self.step = 0


    def forward(self, x):
        # x = B x C x F x F
        T = x.shape[-1]
        if x.dim() == 4:
            mean = torch.mean(x, (1,2,3), keepdim=True)
        else:
            mean = torch.mean(x, (1,2), keepdim=True)
        
        if self.mean is None:
            self.mean = mean
        else:
            alpha = self.step / (self.step + 1)
            self.mean = alpha * self.mean + (1.0 - alpha) * mean
        self.step += 1
        self.step = min(self.step, 80)
        mean = self.mean.detach()
        x /= mean + EPS
        return x
    
    def reset(self):
        self.mean = None
        self.step = 0



class SequenceModel(nn.Module):
    def __init__(
            self,
            input_size,
            output_size,
            hidden_size,
            num_layers,
            bidirectional,
            sequence_model="GRU",
            output_activate_function="Tanh"
    ):
        """
        序列模型，可选 LSTM 或 CRN，支持子带输入

        Args:
            input_size: 每帧输入特征大小
            output_size: 每帧输出特征大小
            hidden_size: 序列模型隐层单元数量
            num_layers:  层数
            bidirectional: 是否为双向
            sequence_model: LSTM | GRU
            output_activate_function: Tanh | ReLU
        """
        super().__init__()
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

        # Activation function layer
        if output_activate_function:
            if output_activate_function == "Tanh":
                self.activate_function = nn.Tanh()
            elif output_activate_function == "ReLU":
                self.activate_function = nn.ReLU()
            elif output_activate_function == "ReLU6":
                self.activate_function = nn.ReLU6()
            elif output_activate_function == "Sigmoid":
                self.activate_function = nn.Sigmoid()
            else:
                raise NotImplementedError(f"Not implemented activation function {self.activate_function}")

        self.output_activate_function = output_activate_function

    def forward(self, x, h0):
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
        o, h = self.sequence_model(x, h0)
        o = self.fc_output_layer(o)
        if self.output_activate_function:
            o = self.activate_function(o)
        o = o.permute(0, 2, 1).contiguous()  # [B, T, F] => [B, F, T]
        return o, h


class BaseModel(nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()

    @staticmethod
    def unfold(input, num_neighbor):
        """
        Along with the frequency dim, split overlapped sub band units from spectrogram.

        Args:
            input: [B, C, F, T]
            num_neighbor:

        Returns:
            [B, N, C, F_s, T], F 为子频带的频率轴大小, e.g. [2, 161, 1, 19, 200]
        """
        assert input.dim() == 4, f"The dim of input is {input.dim()}. It should be four dim."
        batch_size, num_channels, num_freqs, num_frames = input.size()

        if num_neighbor < 1:
            # No change for the input
            return input.permute(0, 2, 1, 3).reshape(batch_size, num_freqs, num_channels, 1, num_frames)

        output = input.reshape(batch_size * num_channels, 1, num_freqs, num_frames)
        sub_band_unit_size = num_neighbor * 2 + 1

        # Pad to the top and bottom
        output = functional.pad(output, [0, 0, num_neighbor, num_neighbor], mode="reflect")

        output = functional.unfold(output, (sub_band_unit_size, num_frames))
        assert output.shape[-1] == num_freqs, f"n_freqs != N (sub_band), {num_freqs} != {output.shape[-1]}"

        # Split the dim of the unfolded feature
        output = output.reshape(batch_size, num_channels, sub_band_unit_size, num_frames, num_freqs)
        output = output.permute(0, 4, 1, 2, 3).contiguous()

        return output

    @staticmethod
    def _reduce_complexity_separately(sub_band_input, full_band_output, device):
        """

        Args:
            sub_band_input: [60, 257, 1, 33, 200]
            full_band_output: [60, 257, 1, 3, 200]
            device:

        Notes:
            1. 255 and 256 freq not able to be trained
            2. batch size 应该被 3 整除，否则最后一部分 batch 内的频率无法很好的训练

        Returns:
            [60, 85, 1, 36, 200]
        """
        batch_size = full_band_output.shape[0]
        n_freqs = full_band_output.shape[1]
        sub_batch_size = batch_size // 3
        final_selected = []

        for idx in range(3):
            # [0, 60) => [0, 20)
            sub_batch_indices = torch.arange(idx * sub_batch_size, (idx + 1) * sub_batch_size, device=device)
            full_band_output_sub_batch = torch.index_select(full_band_output, dim=0, index=sub_batch_indices)
            sub_band_output_sub_batch = torch.index_select(sub_band_input, dim=0, index=sub_batch_indices)

            # Avoid to use padded value (first freq and last freq)
            # i = 0, (1, 256, 3) = [1, 4, ..., 253]
            # i = 1, (2, 256, 3) = [2, 5, ..., 254]
            # i = 2, (3, 256, 3) = [3, 6, ..., 255]
            freq_indices = torch.arange(idx + 1, n_freqs - 1, step=3, device=device)
            full_band_output_sub_batch = torch.index_select(full_band_output_sub_batch, dim=1, index=freq_indices)
            sub_band_output_sub_batch = torch.index_select(sub_band_output_sub_batch, dim=1, index=freq_indices)

            # ([30, 85, 1, 33 200], [30, 85, 1, 3, 200]) => [30, 85, 1, 36, 200]

            final_selected.append(torch.cat([sub_band_output_sub_batch, full_band_output_sub_batch], dim=-2))

        return torch.cat(final_selected, dim=0)

    @staticmethod
    def sband_forgetting_norm(input, train_sample_length):
        """
        与 forgetting norm相同，但使用拼接后模型的中间频带来计算均值
        效果不好
        Args:
            input:
            train_sample_length:

        Returns:

        """
        assert input.ndim == 3
        batch_size, n_freqs, n_frames = input.size()

        eps = 1e-10
        alpha = (train_sample_length - 1) / (train_sample_length + 1)
        mu = 0
        mu_list = []

        for idx in range(input.shape[-1]):
            if idx < train_sample_length:
                alp = torch.min(torch.tensor([(idx - 1) / (idx + 1), alpha]))
                mu = alp * mu + (1 - alp) * torch.mean(input[:, :, idx], dim=1).reshape(batch_size, 1)  # [B, 1]
            else:
                mu = alpha * mu + (1 - alpha) * input[:, (n_freqs // 2 - 1), idx].reshape(batch_size, 1)

            mu_list.append(mu)

            # print("input", input[:, :, idx].min(), input[:, :, idx].max(), input[:, :, idx].mean())
            # print(f"alp {idx}: ", alp)
            # print(f"mu {idx}: {mu[128, 0]}")

        mu = torch.stack(mu_list, dim=-1)  # [B, 1, T]
        input = input / (mu + eps)
        return input

    @staticmethod
    def forgetting_norm(input, sample_length_in_training):
        """
        输入为三维，通过不断估计邻近的均值来作为当前 norm 时的均值

        Args:
            input: [B, F, T]
            sample_length_in_training: 训练时的长度，用于计算平滑因子

        Returns:

        """
        assert input.ndim == 3
        batch_size, n_freqs, n_frames = input.size()
        eps = 1e-10
        mu = 0
        alpha = (sample_length_in_training - 1) / (sample_length_in_training + 1)

        mu_list = []
        for idx in range(input.shape[-1]):
            if idx < sample_length_in_training:
                alp = torch.min(torch.tensor([(idx - 1) / (idx + 1), alpha]))
                mu = alp * mu + (1 - alp) * torch.mean(input[:, :, idx], dim=1).reshape(batch_size, 1)  # [B, 1]
            else:
                current_frame_mu = torch.mean(input[:, :, idx], dim=1).reshape(batch_size, 1)  # [B, 1]
                mu = alpha * mu + (1 - alpha) * current_frame_mu

            mu_list.append(mu)

            # print("input", input[:, :, idx].min(), input[:, :, idx].max(), input[:, :, idx].mean())
            # print(f"alp {idx}: ", alp)
            # print(f"mu {idx}: {mu[128, 0]}")

        mu = torch.stack(mu_list, dim=-1)  # [B, 1, T]
        input = input / (mu + eps)
        return input

    @staticmethod
    def hybrid_norm(input, sample_length_in_training=192):
        """
        Args:
            input: [B, F, T]
            sample_length_in_training:

        Returns:
            [B, F, T]
        """
        assert input.ndim == 3
        device = input.device
        data_type = input.dtype
        batch_size, n_freqs, n_frames = input.size()
        eps = 1e-10

        mu = 0
        alpha = (sample_length_in_training - 1) / (sample_length_in_training + 1)
        mu_list = []
        for idx in range(input.shape[-1]):
            if idx < sample_length_in_training:
                alp = torch.min(torch.tensor([(idx - 1) / (idx + 1), alpha]))
                mu = alp * mu + (1 - alp) * torch.mean(input[:, :, idx], dim=1).reshape(batch_size, 1)  # [B, 1]
                mu_list.append(mu)
            else:
                break
        initial_mu = torch.stack(mu_list, dim=-1)  # [B, 1, T]

        step_sum = torch.sum(input, dim=1)  # [B, T]
        cumulative_sum = torch.cumsum(step_sum, dim=-1)  # [B, T]

        entry_count = torch.arange(n_freqs, n_freqs * n_frames + 1, n_freqs, dtype=data_type, device=device)
        entry_count = entry_count.reshape(1, n_frames)  # [1, T]
        entry_count = entry_count.expand_as(cumulative_sum)  # [1, T] => [B, T]

        cum_mean = cumulative_sum / entry_count  # B, T

        cum_mean = cum_mean.reshape(batch_size, 1, n_frames)  # [B, 1, T]

        # print(initial_mu[0, 0, :50])
        # print("-"*60)
        # print(cum_mean[0, 0, :50])
        cum_mean[:, :, :sample_length_in_training] = initial_mu

        return input / (cum_mean + eps)

    @staticmethod
    def offline_laplace_norm(input):
        """

        Args:
            input: [B, C, F, T]

        Returns:
            [B, C, F, T]
        """
        # utterance-level mu
        mu = torch.mean(input, dim=(1, 2, 3), keepdim=True)

        normed = input / (mu + 1e-5)

        return normed

    @staticmethod
    def cumulative_laplace_norm(input):
        """

        Args:
            input: [B, C, F, T]

        Returns:

        """
        batch_size, num_channels, num_freqs, num_frames = input.size()
        input = input.reshape(batch_size * num_channels, num_freqs, num_frames)

        step_sum = torch.sum(input, dim=1)  # [B * C, F, T] => [B, T]
        cumulative_sum = torch.cumsum(step_sum, dim=-1)  # [B, T]

        entry_count = torch.arange(
            num_freqs,
            num_freqs * num_frames + 1,
            num_freqs,
            dtype=input.dtype,
            device=input.device
        )
        entry_count = entry_count.reshape(1, num_frames)  # [1, T]
        entry_count = entry_count.expand_as(cumulative_sum)  # [1, T] => [B, T]

        cumulative_mean = cumulative_sum / entry_count  # B, T
        cumulative_mean = cumulative_mean.reshape(batch_size * num_channels, 1, num_frames)

        normed = input / (cumulative_mean + EPS)

        return normed.reshape(batch_size, num_channels, num_freqs, num_frames)

    @staticmethod
    def offline_gaussian_norm(input):
        """
        Zero-Norm
        Args:
            input: [B, C, F, T]

        Returns:
            [B, C, F, T]
        """
        mu = torch.mean(input, dim=(2, 3), keepdim=True)
        std = torch.std(input, dim=(2, 3), keepdim=True)

        normed = (input - mu) / (std + 1e-5)

        return normed

    @staticmethod
    def cumulative_layer_norm(input):
        """
        Online zero-norm

        Args:
            input: [B, C, F, T]

        Returns:
            [B, C, F, T]
        """
        batch_size, num_channels, num_freqs, num_frames = input.size()
        input = input.reshape(batch_size * num_channels, num_freqs, num_frames)

        step_sum = torch.sum(input, dim=1)  # [B * C, F, T] => [B, T]
        step_pow_sum = torch.sum(torch.square(input), dim=1)

        cumulative_sum = torch.cumsum(step_sum, dim=-1)  # [B, T]
        cumulative_pow_sum = torch.cumsum(step_pow_sum, dim=-1)  # [B, T]

        entry_count = torch.arange(
            num_freqs,
            num_freqs * num_frames + 1,
            num_freqs,
            dtype=input.dtype,
            device=input.device
        )
        entry_count = entry_count.reshape(1, num_frames)  # [1, T]
        entry_count = entry_count.expand_as(cumulative_sum)  # [1, T] => [B, T]

        cumulative_mean = cumulative_sum / entry_count  # [B, T]
        cumulative_var = (cumulative_pow_sum - 2 * cumulative_mean * cumulative_sum) / entry_count + cumulative_mean.pow(2)  # [B, T]
        cumulative_std = torch.sqrt(cumulative_var + EPS)  # [B, T]

        cumulative_mean = cumulative_mean.reshape(batch_size * num_channels, 1, num_frames)
        cumulative_std = cumulative_std.reshape(batch_size * num_channels, 1, num_frames)

        normed = (input - cumulative_mean) / cumulative_std

        return normed.reshape(batch_size, num_channels, num_freqs, num_frames)

    def norm_wrapper(self, norm_type: str):
        if norm_type == "offline_laplace_norm":
            norm = self.offline_laplace_norm
        elif norm_type == "cumulative_laplace_norm":
            norm = self.cumulative_laplace_norm
        elif norm_type == "offline_gaussian_norm":
            norm = self.offline_gaussian_norm
        elif norm_type == "cumulative_layer_norm":
            norm = self.cumulative_layer_norm
        else:
            raise NotImplementedError("You must set up a type of Norm. "
                                      "e.g. offline_laplace_norm, cumulative_laplace_norm, forgetting_norm, etc.")
        return norm

    def weight_init(self, m):
        """
        Usage:
            model = Model()
            model.apply(weight_init)
        """
        if isinstance(m, nn.Conv1d):
            init.normal_(m.weight.data)
            if m.bias is not None:
                init.normal_(m.bias.data)
        elif isinstance(m, nn.Conv2d):
            init.xavier_normal_(m.weight.data)
            if m.bias is not None:
                init.normal_(m.bias.data)
        elif isinstance(m, nn.Conv3d):
            init.xavier_normal_(m.weight.data)
            if m.bias is not None:
                init.normal_(m.bias.data)
        elif isinstance(m, nn.ConvTranspose1d):
            init.normal_(m.weight.data)
            if m.bias is not None:
                init.normal_(m.bias.data)
        elif isinstance(m, nn.ConvTranspose2d):
            init.xavier_normal_(m.weight.data)
            if m.bias is not None:
                init.normal_(m.bias.data)
        elif isinstance(m, nn.ConvTranspose3d):
            init.xavier_normal_(m.weight.data)
            if m.bias is not None:
                init.normal_(m.bias.data)
        elif isinstance(m, nn.BatchNorm1d):
            init.normal_(m.weight.data, mean=1, std=0.02)
            init.constant_(m.bias.data, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.normal_(m.weight.data, mean=1, std=0.02)
            init.constant_(m.bias.data, 0)
        elif isinstance(m, nn.BatchNorm3d):
            init.normal_(m.weight.data, mean=1, std=0.02)
            init.constant_(m.bias.data, 0)
        elif isinstance(m, nn.Linear):
            init.xavier_normal_(m.weight.data)
            init.normal_(m.bias.data)
        elif isinstance(m, nn.LSTM):
            for param in m.parameters():
                if len(param.shape) >= 2:
                    init.orthogonal_(param.data)
                else:
                    init.normal_(param.data)
        elif isinstance(m, nn.LSTMCell):
            for param in m.parameters():
                if len(param.shape) >= 2:
                    init.orthogonal_(param.data)
                else:
                    init.normal_(param.data)
        elif isinstance(m, nn.GRU):
            for param in m.parameters():
                if len(param.shape) >= 2:
                    init.orthogonal_(param.data)
                else:
                    init.normal_(param.data)
        elif isinstance(m, nn.GRUCell):
            for param in m.parameters():
                if len(param.shape) >= 2:
                    init.orthogonal_(param.data)
                else:
                    init.normal_(param.data)



class FullSubNet(BaseModel):
    def __init__(self,
                 num_freqs,
                 look_ahead,
                 sequence_model,
                 fb_num_neighbors,
                 sb_num_neighbors,
                 fb_output_activate_function,
                 sb_output_activate_function,
                 fb_model_hidden_size,
                 sb_model_hidden_size,
                 num_mics,
                 norm_type="offline_laplace_norm",
                 num_groups_in_drop_band=2,
                 num_layers=2,
                 weight_init=True,
                 sample_rate=16000, 
                 segment_length=400,
                 win_length=20, 
                 hop_length=10, 
                 n_fft=320
                 ):
        """
        FullSubNet model (cIRM mask)

        Args:
            num_freqs: Frequency dim of the input
            look_ahead: Number of use of the future frames
            fb_num_neighbors: How much neighbor frequencies at each side from fullband model's output
            sb_num_neighbors: How much neighbor frequencies at each side from noisy spectrogram
            sequence_model: Chose one sequence model as the basic model e.g., GRU, LSTM
            fb_output_activate_function: fullband model's activation function
            sb_output_activate_function: subband model's activation function
            norm_type: type of normalization, see more details in "BaseModel" class
        """
        super().__init__()
        assert sequence_model in ("GRU", "LSTM"), f"{self.__class__.__name__} only support GRU and LSTM."

        self.stft = STFT(sample_rate=sample_rate, win_length=win_length, 
                        hop_length=hop_length, n_fft=n_fft)

        self.istft = ISTFT(sample_rate=sample_rate, win_length=win_length, 
                        hop_length=hop_length, n_fft=n_fft)

        self.fb_model = SequenceModel(
            input_size=num_freqs * num_mics,
            output_size=num_freqs,
            hidden_size=fb_model_hidden_size,
            num_layers=num_layers,
            bidirectional=False,
            sequence_model=sequence_model,
            output_activate_function=fb_output_activate_function
        )

        self.sb_model = SequenceModel(
            input_size=(sb_num_neighbors * 2 + 1) + (fb_num_neighbors * 2 + 1),
            output_size=2,
            hidden_size=sb_model_hidden_size,
            num_layers=num_layers,
            bidirectional=False,
            sequence_model=sequence_model,
            output_activate_function=sb_output_activate_function
        )

        self.sb_num_neighbors = sb_num_neighbors
        self.fb_num_neighbors = fb_num_neighbors
        self.look_ahead = look_ahead
        self.fb_model_hidden_size = fb_model_hidden_size
        self.sb_model_hidden_size = sb_model_hidden_size
        self.num_layers = num_layers
        self.num_mics = num_mics
        self.num_freqs = num_freqs
        self.segment_length = segment_length
        self.norm_fb = CumLayerNorm()
        self.norm_sb = CumLayerNorm()
        self.num_groups_in_drop_band = num_groups_in_drop_band

        self.fh = None
        self.sh = None
        self.exist_prob = None

        if weight_init:
            self.apply(self.weight_init)

    def forward(self, noisy_complex):
        """
        Args:
            noisy: noisy magnitude spectrogram

        Returns:
            The real part and imag part of the enhanced spectrogram

        Shapes:
            noisy: [B, M, F, N*T]
            return: [B, 2, F, N*T]
        """
        assert noisy_complex.dim() == 4
        noisy = torch.sqrt(noisy_complex[:,:self.num_mics]**2 + noisy_complex[:,self.num_mics:]**2 + EPS)
        #noisy = functional.pad(noisy, [0, self.look_ahead])  # Pad the look ahead
        batch_size, num_channels, num_freqs, num_frames = noisy.size()
        #assert num_channels == 1, f"{self.__class__.__name__} takes the complex feature as inputs."

        # Fullband model
        fb_input = self.norm_fb(noisy).reshape(batch_size, num_channels * num_freqs, num_frames)
        fb_output, fh = self.fb_model(fb_input, self.fh)
        fb_output = fb_output.unsqueeze(1)
        # Unfold fullband model's output, [B, N=F, F_f, T]. N is the number of sub-band units
        fb_output_unfolded = self.unfold(fb_output, num_neighbor=self.fb_num_neighbors)
        fb_output_unfolded = fb_output_unfolded.reshape(batch_size, num_freqs, (self.fb_num_neighbors * 2 + 1), num_frames)

        # Unfold noisy spectrogram, [B, N=F, F_s, T]
        noisy_unfolded = self.unfold(functional.pad(noisy[:,0], [0, self.look_ahead]).unsqueeze(1), num_neighbor=self.sb_num_neighbors)
        noisy_unfolded = noisy_unfolded.reshape(batch_size, num_freqs, (self.sb_num_neighbors * 2 + 1), num_frames)

        # Concatenation, [B, F, (F_s + F_f), T]
        sb_input = torch.cat([noisy_unfolded, fb_output_unfolded], dim=2)
        sb_input = self.norm_sb(sb_input)

        # Speeding up training without significant performance degradation.
        # These will be updated to the paper later.
        # if batch_size > 1:
        #     sb_input = drop_band(sb_input.permute(0, 2, 1, 3), num_groups=self.num_groups_in_drop_band)  # [B, C*(F_s + F_f), F//num_groups, T]
        #     num_freqs = sb_input.shape[2]
        #     sb_input = sb_input.permute(0, 2, 1, 3)  # [B, F//num_groups, C*(F_s + F_f), T]

        sb_input = sb_input.reshape(
            batch_size * num_freqs,
            (self.sb_num_neighbors * 2 + 1) + (self.fb_num_neighbors * 2 + 1),
            num_frames
        )
        # [B * F, (F_s + F_f), T] => [B * F, 2, T] => [B, 2, F, T]
        sb_mask, sh = self.sb_model(sb_input, self.sh)
        sb_mask = sb_mask.reshape(batch_size, num_freqs, 2, num_frames).permute(0, 2, 1, 3).contiguous()
        
        self.fh = (fh[0].detach(), fh[1].detach())
        self.sh = (sh[0].detach(), sh[1].detach())
        #[B, 2, F, T]
        #output = sb_mask[:, :, :, self.look_ahead:]
        output = sb_mask
        return output
    
    def reset_state(self, batch_size, dtype, device):
        self.fh = (torch.zeros(self.num_layers, batch_size, self.fb_model_hidden_size, dtype=dtype, device=device), 
                torch.zeros(self.num_layers, batch_size, self.fb_model_hidden_size,dtype=dtype, device=device))
        self.sh = (torch.zeros(self.num_layers, self.num_freqs*batch_size, self.sb_model_hidden_size,dtype=dtype, device=device), 
                torch.zeros(self.num_layers, self.num_freqs*batch_size, self.sb_model_hidden_size,dtype=dtype, device=device))
        self.norm_fb.reset()
        self.norm_sb.reset()


    def stft_trans(self, x):
        #B*N, M, K
        B, M, L = x.shape
        #B*N*M, L
        x = x.reshape(-1,L)
        #B*N, M, F, T, complex 
        x = self.stft(x).reshape(B,M,-1,self.num_freqs,2).transpose(2,3)
        #B*N, 2*M, F, T
        x = torch.cat(torch.split(x,1,dim=-1),dim=1).squeeze(-1)
        return x
    
    def istft_trans(self, x):
        B, F, T, _ = x.shape
        #B, T, F, 2
        x = x.permute(0,2,1,3)
        #B, L 
        x = self.istft(x)
        return x
    
    def segmentation(self, x):
        x, gap = segmentation(x, self.segment_length)
        return x, gap
    
    def overadd(self, x, gap):
        #B, N, K
        x = over_add(x, gap)
        return x

    
    def preprocessing(self, mixture, source=None):
        """
        mixture, source, noise: B, M, L
        output: N, B, 2*M, F, T
        """
        batch_size = len(mixture)
        #B*N, M, K
        seg_x, gap = self.segmentation(mixture)
        #B*N, 2*M, F, T
        x = self.stft_trans(seg_x)
        #N, B, 2*M, F, T
        x = torch.stack(torch.split(x,len(x) // batch_size,dim=0),dim=1)
        
        if source is not None:
            #B*N, M, K
            seg_s, gap = self.segmentation(source)
            s = self.stft_trans(seg_s)
            #N, B, 2*M, F, T
            s = torch.stack(torch.split(s,len(s) // batch_size,dim=0),dim=1)
            #source only return the first channel
            s = torch.stack([s[:,:,0], s[:,:,self.num_mics]], dim=2)
            return x, s, gap
        else:
            return x, None,  gap
    
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
       
    
    def realtime_process(self, mixture, source = None, flag=False, train=True):
        B, C, T = mixture.shape
        if not flag:
            pad = torch.zeros((B, C, self.segment_length // 2), dtype=mixture.dtype, device=mixture.device)
            mixture = torch.cat([pad, mixture], dim=-1)
            source = torch.cat([pad, source], dim=-1)
        x, s, gap = self.preprocessing(mixture, source)
        memory_size = 0
        #N, B, C, F, T
        num_segments, B, C, F, T = x.shape
        if not flag:
            self.reset_state(B, x.dtype, x.device)
            self.exist_prob = 0.0
        
        exist_prob = self.exist_prob
        pred_crm = torch.zeros((num_segments, B, 2, F, T), dtype=x.dtype, device=x.device)
        win = torch.hann_window(T, device=x.device).reshape(1, 1, 1, 1, T).expand_as(pred_crm)[0]
        
        if train:
            #B, C, F, N*T
            xf = torch.cat(torch.split(x, 1, dim=0), dim=-1).squeeze(0).contiguous()
            #B, 2, F, N*T
            pred_crm = self.forward(xf)
            #N, B, 2, F, T
            pred_crm = torch.stack(torch.split(pred_crm, T, dim=-1), dim=0).contiguous()
        else:
            # xf = torch.sqrt(x[:,:,0]**2 + x[:,:,self.num_mics]**2 + EPS)
            # sf = torch.sqrt(s[:,:,0]**2 + s[:,:,1]**2 + EPS)
            #N, B, C, F, T
            for idx in range(num_segments):
                # #计算语音存在概率
                # c = 0.02
                # nf = xf[idx] - sf[idx]
                # nf2 = torch.mean(nf**2, dim=(0,2))
                # xf2 = torch.mean(xf[idx]**2, dim=(0,2))
                # sf2 = torch.mean(sf[idx]**2, dim=(0,2))
                # delta = (nf2 + sf2) / (nf2 + EPS) * torch.exp(-xf2 / (nf2 + sf2 + EPS) + xf2 / (nf2 + EPS))
                # prob = torch.tanh(torch.mean(torch.log(delta)))
                # exist_prob = c * exist_prob + (1.0-c) * prob
                inp = x[idx]
                with autocast():
                    crm = self.forward(inp)
                pred_crm[idx] = crm[..., -T:]
        # print("Prob: ",exist_prob)
        #[N,B,2,F,T]
        self.exist_prob = exist_prob
        crm = decompress_cIRM(pred_crm)
        x = torch.stack([x[:,:,0], x[:,:,self.num_mics]],dim=2)
        enhanced_real = crm[:,:,0] * x[:,:,0] - crm[:,:,1] * x[:,:,1]
        enhanced_imag = crm[:,:,1] * x[:,:,0] + crm[:,:,0] * x[:,:,1]
        pred_source = torch.stack((enhanced_real, enhanced_imag), dim=2)
        #N, B, 2, F, T -> B, L
        pred_source = self.postprocessing(pred_source.permute(0, 1, 3, 4, 2).contiguous(), gap)
        if not flag:
            pred_source = pred_source[..., self.segment_length // 2: ]
        if source is None:
            return pred_source
        else:
            return pred_source, pred_crm, s, x

    
    def compute_loss(self, source, pred_source, xf, sf, cIRM, length):
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
        mae = stoi_loss(source, pred_source, length)

        sisnr = -cal_si_snr(pred_source, source, length) 
        loss = 0.7 * mae + 0.3 * sisnr
        print(sisnr, '\n')
        if torch.isnan(loss):
            mae = mae.fill_(0.0)
            sisnr = sisnr.fill_(0.0)
            loss = loss.fill_(0.0)
        return loss, mae, sisnr


if __name__ == "__main__":
    import time

    with torch.no_grad():
        model = FullSubNet(
            sb_num_neighbors=15,
            fb_num_neighbors=0,
            num_freqs=241,
            look_ahead=0,
            sequence_model="LSTM",
            fb_output_activate_function="ReLU",
            sb_output_activate_function=None,
            fb_model_hidden_size=512,
            sb_model_hidden_size=384,
            num_layers= 2,
            weight_init=True,
            num_mics = 3,
            norm_type="cumulative_layer_norm",
            num_groups_in_drop_band=2,
            segment_length=960,
            sample_rate=16000, 
            win_length=30, 
            hop_length=15, 
            n_fft=480
        )
        ipt = torch.rand(1, 3, 64000)  # 1.6s
        f = (torch.zeros(2, 1, 512), torch.zeros(2, 1, 512))
        s = (torch.zeros(2, 161, 384), torch.zeros(2, 161, 384))
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
        
        pred_source, pred_crm, s, xf = model.realtime_process(ipt, ipt)
        print(f"Real Time Process for {ipt.shape[-1]/16000}s: {time.time() - start}")
        print(pred_source.shape, pred_crm.shape, s.shape, xf.shape)
        
        parameters = sum(param.numel() for param in model.parameters())
        print(str(parameters / 10**6) +' MB ')