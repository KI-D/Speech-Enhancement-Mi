import numpy as np
from itertools import permutations
from torch.autograd import Variable
import torch.nn.functional as F
import scipy,time,numpy
import torch

EPS = 1e-8

# original implementation
def compute_measures(se,s,j):
    Rss=s.transpose().dot(s)
    this_s=s[:,j]

    a=this_s.transpose().dot(se)/Rss[j,j]
    e_true=a*this_s
    e_res=se-a*this_s
    Sss=np.sum((e_true)**2)
    Snn=np.sum((e_res)**2)

    SDR=10*np.log10(Sss/Snn)

    Rsr= s.transpose().dot(e_res)
    b=np.linalg.inv(Rss).dot(Rsr)

    e_interf = s.dot(b)
    e_artif= e_res-e_interf

    SIR=10*np.log10(Sss/np.sum((e_interf)**2))
    SAR=10*np.log10(Sss/np.sum((e_artif)**2))
    return SDR, SIR, SAR

def GetSDR(se,s):
    se=se-np.mean(se,axis=0)
    s=s-np.mean(s,axis=0)
    nsampl,nsrc=se.shape
    nsampl2,nsrc2=s.shape
    assert(nsrc2==nsrc)
    assert(nsampl2==nsampl)

    SDR=np.zeros((nsrc,nsrc))
    SIR=SDR.copy()
    SAR=SDR.copy()

    for jest in range(nsrc):
        for jtrue in range(nsrc):
            SDR[jest,jtrue],SIR[jest,jtrue],SAR[jest,jtrue]=compute_measures(se[:,jest],s,jtrue)


    perm=list(permutations(np.arange(nsrc)))
    nperm=len(perm)
    meanSIR=np.zeros((nperm,))
    for p in range(nperm):
        tp=SIR.transpose().reshape(nsrc*nsrc)
        idx=np.arange(nsrc)*nsrc+list(perm[p])
        meanSIR[p]=np.mean(tp[idx])
    popt=np.argmax(meanSIR)
    per=list(perm[popt])
    idx=np.arange(nsrc)*nsrc+per
    SDR=SDR.transpose().reshape(nsrc*nsrc)[idx]
    SIR=SIR.transpose().reshape(nsrc*nsrc)[idx]
    SAR=SAR.transpose().reshape(nsrc*nsrc)[idx]
    return SDR, SIR, SAR, per

# Pytorch implementation with batch processing
def calc_sdr_torch(estimation, origin, mask=None):
    """
    batch-wise SDR caculation for one audio file on pytorch Variables.
    estimation: (batch, nsample)
    origin: (batch, nsample)
    mask: an optional mask for sequence masking. This is for cases where zero-padding was applied at the end and should not be consider for SDR calculation.
    """
    
    estimation = estimation - torch.mean(estimation, 1, keepdim=True)
    origin = origin - torch.mean(origin, 1, keepdim=True)

    if mask is not None:
        origin = origin * mask
        estimation = estimation * mask
        
    def calculate(estimation, origin):
        origin_power = torch.pow(origin, 2).sum(1, keepdim=True) + EPS  # (batch, 1)
        scale = torch.sum(origin*estimation, 1, keepdim=True) / origin_power  # (batch, 1)

        est_true = scale * origin  # (batch, nsample)
        est_res = estimation - est_true  # (batch, nsample)

        true_power = torch.pow(est_true, 2).sum(1) + EPS
        res_power = torch.pow(est_res, 2).sum(1) + EPS

        return 10*torch.log10(true_power) - 10*torch.log10(res_power)  # (batch, )
        
    best_sdr = calculate(estimation, origin)
    
    return best_sdr


def batch_SDR_torch(estimation, origin, mask=None, return_perm=False):
    """
    batch-wise SDR caculation for multiple audio files.
    estimation: (batch, nsource, nsample)
    origin: (batch, nsource, nsample)
    mask: optional, (batch, nsample), binary
    return_perm: bool, whether to return the permutation index. Default is false.
    """
    
    batch_size_est, nsource_est, nsample_est = estimation.size()
    batch_size_ori, nsource_ori, nsample_ori = origin.size()
    
    assert batch_size_est == batch_size_ori, "Estimation and original sources should have same shape."
    assert nsource_est == nsource_ori, "Estimation and original sources should have same shape."
    assert nsample_est == nsample_ori, "Estimation and original sources should have same shape."
    
    assert nsource_est < nsample_est, "Axis 1 should be the number of sources, and axis 2 should be the signal."
    
    batch_size = batch_size_est
    nsource = nsource_est
    
    # zero mean signals
    estimation = estimation - torch.mean(estimation, 2, keepdim=True).expand_as(estimation)
    origin = origin - torch.mean(origin, 2, keepdim=True).expand_as(estimation)
    
    # SDR for each permutation
    SDR = torch.zeros((batch_size, nsource, nsource)).type(estimation.type())
    for i in range(nsource):
        for j in range(nsource):
            SDR[:,i,j] = calc_sdr_torch(estimation[:,i], origin[:,j], mask)
    
    # choose the best permutation
    SDR_max = []
    SDR_perm = []
    perm = sorted(list(set(permutations(np.arange(nsource)))))
    for permute in perm:
        sdr = []
        for idx in range(len(permute)):
            sdr.append(SDR[:,idx,permute[idx]].view(batch_size,-1))
        sdr = torch.sum(torch.cat(sdr, 1), 1)
        SDR_perm.append(sdr.view(batch_size, 1))
    SDR_perm = torch.cat(SDR_perm, 1)
    SDR_max, SDR_idx = torch.max(SDR_perm, dim=1)
    
    if not return_perm:
        return SDR_max / nsource
    else:
        return SDR_max / nsource, SDR_idx


def worker_fn(worker_id):
    seed = 2021
    seed += worker_id
    np.random.seed(seed)


def collate_fn(batch, pad_index=0):
    mix = pad_sequence([b['mix'].squeeze() for b in batch]).float()
    source = pad_sequence([b['source'].squeeze() for b in batch]).float()
    if source.dim()<=4:
        source = source.unsqueeze(1)
    noise = pad_sequence([b['noise'].squeeze() for b in batch]).float()
    length = torch.stack([b['length'] for b in batch], dim=0)
    #spk = torch.stack([b['spk'] for b in batch], dim=0)
    batch = {'mix':mix, 'source':source, 'noise':noise, 'length':length, 'flag':batch[0]['flag']}
    return batch


def pad_sequence(source_list, pad_value = 0):
    '''
    source_list: List of tensor(chan, Tc) or (src, chan, Tc)
    '''
    length = [x.shape[-1] for x in source_list]
    max_length = max(length)
    trans = False
    if source_list[0].dim() >= 3:
        trans = True
        n_spk, n_chan, _ = source_list[0].shape
    for i, source in enumerate(source_list):
        if trans:
            source = source.reshape(n_spk*n_chan, -1)
        source = F.pad(source, (0,max_length-length[i],0,0), "constant", value=pad_value)
        if trans:
            source = source.reshape(n_spk, n_chan, -1)
        source_list[i] = source
    source = torch.stack(source_list, dim=0)
    return source

def get_mask(source, length):
    """
    Args:
        source: [B, C, T] or [B, T]
        length: [B]
    Returns:
        mask: [B, 1, T]
    """
    if source.dim()==3:
        B, _, T = source.size()
        mask = source.new_ones((B, 1, T))
        for i in range(B):
            mask[i, :, length[i]:] = 0
    else:
        B, T = source.size()
        mask = source.new_ones((B, T))
        for i in range(B):
            mask[i, length[i]:] = 0
    return mask


def cal_si_snr(separated, source, length=None, eps=1e-8):
    def l2norm(mat, keepdim=False):
        return torch.norm(mat, dim=-1, keepdim=keepdim)
    B = len(separated)
    sisnr = 0.0
    for i in range(B):
        if length is None:
            separated_i = separated[i]
            source_i = source[i]
        else:
            separated_i = separated[i, :length[i]]
            source_i = source[i, :length[i]]
        separated_i = separated_i - torch.mean(separated_i, dim=-1, keepdim=True)
        source_i = source_i - torch.mean(source_i, dim=-1, keepdim=True)
        true = torch.sum(separated_i * source_i, dim=-1, keepdim=True) * source_i / (l2norm(source_i, keepdim=True)**2+eps)
        sisnr += 20*torch.log10(eps + l2norm(true)/(l2norm(separated_i-true)+eps))
    return sisnr / B


def cal_si_snr_with_pit(separated, source, length):
    """Calculate SI-SNR with PIT training.
    Args:
        separated: [B, N, T]
        source: [B, N, T], B is batch size
        length: [B], each item is between [0, T]
    """
    assert source.size() == separated.size()
    B, N, T = source.size()
    # mask padding position along T
    mask = get_mask(source, length)
    separated *= mask

    # Step 1. Zero-mean norm
    num_samples = length.view(-1, 1, 1).float()  # [B, 1, 1]
    mean_target = torch.sum(source, dim=-1, keepdim=True) / num_samples
    mean_separated = torch.sum(separated, dim=-1, keepdim=True) / num_samples
    zero_mean_target = source - mean_target
    zero_mean_separated = separated - mean_separated
    # mask padding position along T
    zero_mean_target *= mask
    zero_mean_separated *= mask

    # Step 2. SI-SNR with PIT
    # reshape to use broadcast
    s_target = torch.unsqueeze(zero_mean_target, dim=1)  # [B, 1, N, T]
    s_separated = torch.unsqueeze(zero_mean_separated, dim=2)  # [B, N, 1, T]
    # s_target = <s', s>s / ||s||^2
    pair_wise_dot = torch.sum(s_separated * s_target, dim=3, keepdim=True)  # [B, N, N, 1]
    s_target_energy = torch.sum(s_target ** 2, dim=3, keepdim=True) + EPS  # [B, 1, N, 1]
    pair_wise_proj = pair_wise_dot * s_target / s_target_energy  # [B, N, N, T]
    # e_noise = s' - s_target
    e_noise = s_separated - pair_wise_proj  # [B, N, N, T]
    # SI-SNR = 10 * log_10(||s_target||^2 / ||e_noise||^2)
    pair_wise_si_snr = torch.sum(pair_wise_proj ** 2, dim=3) / (torch.sum(e_noise ** 2, dim=3) + EPS)
    pair_wise_si_snr = 10 * torch.log10(pair_wise_si_snr + EPS)  # [B, N, N]

    # Get max_snr of each utterance
    # permutations, [N!, N]
    perms = source.new_tensor(list(permutations(range(N))), dtype=torch.long)
    # one-hot, [N!, N, N]
    index = torch.unsqueeze(perms, 2)
    perms_one_hot = source.new_zeros((*perms.size(), N)).scatter_(2, index, 1)
    # [B, N!] <- [B, N, N] einsum [N!, N, N], SI-SNR sum of each permutation
    snr_set = torch.einsum('bij,pij->bp', [pair_wise_si_snr, perms_one_hot])
    max_snr_idx = torch.argmax(snr_set, dim=1)  # [B]
    # max_snr = torch.gather(snr_set, 1, max_snr_idx.view(-1, 1))  # [B, 1]
    max_snr, _ = torch.max(snr_set, dim=1, keepdim=True)
    max_snr /= N
    return max_snr, perms, max_snr_idx


def reorder_source(source, perms, max_snr_idx):
    """
    Args:
        source: [B, N, T]
        perms: [N!, N], permutations
        max_snr_idx: [B], each item is between [0, N!)
    Returns:
        reorder_source: [B, N, T]
    """
    B, N, *_ = source.size()
    # [B, N], permutation whose SI-SNR is max of each utterance
    # for each utterance, reorder estimate source according this permutation
    max_snr_perm = torch.index_select(perms, dim=0, index=max_snr_idx)
    # print('max_snr_perm', max_snr_perm)
    # maybe use torch.gather()/index_select()/scatter() to impl this?
    reorder_source = torch.zeros_like(source)
    for b in range(B):
        reorder_source[b, :] = source[b, max_snr_perm[b]]
    return reorder_source


def pit_sisnr(separated, source, length):
    """
    Args:
        source: [B, N, T], B is batch size
        separated: [B, N, T]
        length: [B]
    """
    max_snr, perms, max_snr_idx = cal_si_snr_with_pit(source, separated, length)
    loss = -torch.mean(max_snr)
    return loss, perm



def padding(input, K):
    """Padding the audio times.

    Arguments
    ---------
    K : int
        Chunks of length.
    P : int
        Hop size.
    input : torch.Tensor
        Tensor of size [B, C, L].
        where, B = Batchsize,
                C = channel,
                L = time points
    """
    B, C, L = input.shape
    P = K // 2
    gap = K - (P + L % K) % K
    if gap > 0:
        pad = torch.zeros(B, C, gap).type(input.type()).to(input.device)
        input = torch.cat([input, pad], dim=-1)

    _pad = torch.zeros(B, C, P).type(input.type()).to(input.device)
    input = torch.cat([_pad, input, _pad], dim=-1)
    return input, gap


def segmentation(input, K):
    """The segmentation stage splits

    Arguments
    ---------
    K : int
        Length of the chunks.
    input : torch.Tensor
        Tensor with dim [B, C, L].

    Return
    -------
    output : torch.tensor
        Tensor with dim [B, C, K].
        where, B = Batchsize,
            C = channel,
            K = time points in each chunk
    """
    B, C, L = input.shape
    #C, B , L
    input = input.permute(1,0,2)
    P = K // 2
    input, gap = padding(input, K)
    # [C, B*N, K]
    input1 = input[..., :-P].reshape(C, -1, K)
    input2 = input[..., P:].reshape(C, -1, K)
    # [B*N, C, K]
    input = (
        torch.cat([input1, input2], dim=-1).reshape(C, -1, K)
    ).permute(1,0,2)

    return input, gap


def over_add(input, gap):
    """Merge the sequence with the overlap-and-add method.

    Arguments
    ---------
    input : torch.tensor
        Tensor with dim [C, N, K].
    gap : int
        Padding length.

    Return
    -------
    output : torch.tensor
        Tensor with dim [C, L].
        where, N = number of chunks 
            C = number of speakers
            K = time points in each chunk
            L = the number of time points

    """
    C, N, K = input.shape
    P = K // 2
    input = input.reshape(C, -1, K * 2)

    input1 = input[:, :, :K].reshape(C, -1)[:, P:]
    input2 = input[:, :, K:].reshape(C, -1)[:, :-P]
    input = (input1 + input2) / 2
    # [C, L]
    if gap > 0:
        input = input[:, :-gap]
    return input


def build_complex_ideal_ratio_mask(noisy, clean) -> torch.Tensor:
    """

    Args:
        noisy: [N, B, 2, F, T]
        clean: [N, B, 2, F, T]

    Returns:
        [N, B, 2, F, T]
    """
    denominator = torch.square(noisy[:,:,0]) + torch.square(noisy[:,:,1]) + EPS

    mask_real = (noisy[:,:,0] * clean[:,:,0] + noisy[:,:,1] * clean[:,:,1]) / denominator
    mask_imag = (noisy[:,:,0] * clean[:,:,1] - noisy[:,:,1] * clean[:,:,0]) / denominator

    complex_ratio_mask = torch.stack((mask_real, mask_imag), dim=2)

    return compress_cIRM(complex_ratio_mask, K=10, C=0.1)


def compress_cIRM(mask, K=10, C=0.1):
    """
        Compress from (-inf, +inf) to [-K ~ K]
    """
    if torch.is_tensor(mask):
        mask = -100 * (mask <= -100) + mask * (mask > -100)
        mask = K * (1 - torch.exp(-C * mask)) / (1 + torch.exp(-C * mask))
    else:
        mask = -100 * (mask <= -100) + mask * (mask > -100)
        mask = K * (1 - np.exp(-C * mask)) / (1 + np.exp(-C * mask))
    return mask


def decompress_cIRM(mask, K=10, limit=9.9):
    mask = limit * (mask >= limit) - limit * (mask <= -limit) + mask * (torch.abs(mask) < limit)
    mask = -K * torch.log((K - mask) / (K + mask))
    return mask


def complex_mul(noisy_r, noisy_i, mask_r, mask_i):
    r = noisy_r * mask_r - noisy_i * mask_i
    i = noisy_r * mask_i + noisy_i * mask_r
    return r, i


def tailor_dB_FS(y, target_dB_FS=-25, eps=1e-6):
    rms = torch.sqrt(torch.mean(y ** 2))
    scalar = 10 ** (target_dB_FS / 20) / (rms + eps)
    y *= scalar
    return y, rms, scalar




# ################################
# From paper: "End-to-End Waveform Utterance Enhancement for Direct Evaluation
# Metrics Optimization by Fully Convolutional Neural Networks", TASLP, 2018
# Authors: Szu-Wei, Fu 2020
# ################################

import torch
import torchaudio
import numpy as np
from speechbrain.utils.torch_audio_backend import get_torchaudio_backend

torchaudio_backend = get_torchaudio_backend()
torchaudio.set_audio_backend(torchaudio_backend)
smallVal = np.finfo("float").eps  # To avoid divide by zero


def thirdoct(fs, nfft, num_bands, min_freq):
    """Returns the 1/3 octave band matrix.

    Arguments
    ---------
    fs : int
        Sampling rate.
    nfft : int
        FFT size.
    num_bands : int
        Number of 1/3 octave bands.
    min_freq : int
        Center frequency of the lowest 1/3 octave band.

    Returns
    -------
    obm : tensor
        Octave Band Matrix.
    """

    f = torch.linspace(0, fs, nfft + 1)
    f = f[: int(nfft / 2) + 1]
    k = torch.from_numpy(np.array(range(num_bands)).astype(float))
    cf = torch.pow(2.0 ** (1.0 / 3), k) * min_freq
    freq_low = min_freq * torch.pow(2.0, (2 * k - 1) / 6)
    freq_high = min_freq * torch.pow(2.0, (2 * k + 1) / 6)
    obm = torch.zeros(num_bands, len(f))  # a verifier

    for i in range(len(cf)):
        # Match 1/3 oct band freq with fft frequency bin
        f_bin = torch.argmin(torch.square(f - freq_low[i]))
        freq_low[i] = f[f_bin]
        fl_ii = f_bin
        f_bin = torch.argmin(torch.square(f - freq_high[i]))
        freq_high[i] = f[f_bin]
        fh_ii = f_bin
        # Assign to the octave band matrix
        obm[i, fl_ii:fh_ii] = 1
    return obm


def removeSilentFrames(x, y, dyn_range=40, N=256, K=128):
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

    Y1 = y[0 : int(y.shape[0]) // N * N].reshape(int(y.shape[0]) // N, N).T
    Y2 = (
        y[128 : (int(y.shape[0]) - 128) // N * N + 128]
        .reshape((int(y.shape[0]) - 128) // N, N)
        .T
    )
    Y = torch.zeros(N, Y1.shape[1] + Y2.shape[1])
    Y[:, 0::2] = Y1
    Y[:, 1::2] = Y2

    x_sil = w.T.repeat(1, X[:, msk].shape[-1]) * X[:, msk]
    y_sil = w.T.repeat(1, X[:, msk].shape[-1]) * Y[:, msk]
    
    x_sil = torch.cat(
        (
            x_sil[0:128, 0],
            (x_sil[0:128, 1:] + x_sil[128:, 0:-1]).T.flatten(),
            x_sil[128:256, -1],
        ),
        axis=0,
    )
    y_sil = torch.cat(
        (
            y_sil[0:128, 0],
            (y_sil[0:128, 1:] + y_sil[128:, 0:-1]).T.flatten(),
            y_sil[128:256, -1],
        ),
        axis=0,
    )

    return [x_sil, y_sil]


def stoi_loss(y_pred_batch, y_true_batch, lens, reduction="mean"):
    """Compute the STOI score and return -1 * that score.

    This function can be used as a loss function for training
    with SGD-based updates.

    Arguments
    ---------
    y_pred_batch : torch.Tensor
        The degraded (enhanced) waveforms.
    y_true_batch : torch.Tensor
        The clean (reference) waveforms.
    lens : torch.Tensor
        The relative lengths of the waveforms within the batch.
    reduction : str
        The type of reduction ("mean" or "batch") to use.

    Example
    -------
    >>> a = torch.sin(torch.arange(16000, dtype=torch.float32)).unsqueeze(0)
    >>> b = a + 0.001
    >>> -stoi_loss(b, a, torch.ones(1))
    tensor(0.7...)
    """
    y_pred_batch = y_pred_batch.cpu()

    y_pred_batch = torch.squeeze(y_pred_batch, dim=-1)
    y_true_batch = torch.squeeze(y_true_batch, dim=-1)

    batch_size = y_pred_batch.shape[0]

    fs = 16000  # Sampling rate
    N = 30  # length of temporal envelope vectors
    J = 15.0  # Number of one-third octave bands

    octave_band = thirdoct(fs=10000, nfft=512, num_bands=15, min_freq=150)
    c = 5.62341325  # 10^(-Beta/20) with Beta = -15
    D = torch.zeros(batch_size)
    device = y_true_batch.device
    resampler = torchaudio.transforms.Resample(fs, 10000)
    for i in range(0, batch_size):  # Run over mini-batches
        y_true = y_true_batch[i, 0 : int(lens[i])].cpu()
        y_pred = y_pred_batch[i, 0 : int(lens[i])].cpu()

        y_true, y_pred = resampler(y_true).to(device), resampler(y_pred).to(device)
        try:
            [y_sil_true, y_sil_pred] = removeSilentFrames(y_true, y_pred)
        except:
            y_sil_true = y_true.cpu()
            y_sil_pred = y_pred.cpu()
        stft_true = torchaudio.transforms.Spectrogram(
            n_fft=512, win_length=256, hop_length=128, power=2
        )(y_sil_true)
        stft_pred = torchaudio.transforms.Spectrogram(
            n_fft=512, win_length=256, hop_length=128, power=2
        )(y_sil_pred)

        OCT_true = torch.sqrt(torch.matmul(octave_band, stft_true) + 1e-14)
        OCT_pred = torch.sqrt(torch.matmul(octave_band, stft_pred) + 1e-14)

        M = int(OCT_true.shape[-1] - (N - 1))  # number of temporal envelope vectors
        if M <= 0:
            X = OCT_true
            Y = OCT_pred
            M = 1
        else:
            X = torch.zeros(15 * M, 30)
            Y = torch.zeros(15 * M, 30)
            for m in range(0, M):  # Run over temporal envelope vectors
                X[m * 15 : (m + 1) * 15, :] = OCT_true[:, m : m + N]
                Y[m * 15 : (m + 1) * 15, :] = OCT_pred[:, m : m + N]


        alpha = torch.norm(X, dim=-1, keepdim=True) / (
            torch.norm(Y, dim=-1, keepdim=True) + smallVal
        )

        ay = Y * alpha
        y = torch.min(ay, X + X * c)

        xn = X - torch.mean(X, dim=-1, keepdim=True)
        xn = xn / (torch.norm(xn, dim=-1, keepdim=True) + smallVal)


        yn = y - torch.mean(y, dim=-1, keepdim=True)
        yn = yn / (torch.norm(yn, dim=-1, keepdim=True) + smallVal)
        d = torch.sum(xn * yn)
        D[i] = d / (J * M)

    if reduction == "mean":
        return -D.mean()

    return -D
