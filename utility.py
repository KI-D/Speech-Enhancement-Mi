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


def median_filter(x, kernel_size = 3):
    y = torch.zeros_like(x)
    for i in range(len(x) + kernel_size // 2):
        y[i] = torch.median(x[i-kernel_size//2 : i+kernel_size//2])
    return y

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

def kldiv_loss(log_probabilities, targets, length=None, label_smoothing=0.0, allowed_len_diff=3, pad_idx=0, reduction="mean"):
    '''
    log_probabilities: [B, T, K]
    targets: [B, T]
    '''

    if log_probabilities.dim() == 2:
        log_probabilities = log_probabilities.unsqueeze(1)

    bz, time, n_class = log_probabilities.shape
    targets = targets.long().detach()

    confidence = 1 - label_smoothing

    log_probabilities = log_probabilities.view(-1, n_class)
    targets = targets.view(-1)
    with torch.no_grad():
        true_distribution = log_probabilities.clone()
        true_distribution.fill_(label_smoothing / (n_class - 1))
        ignore = targets == pad_idx
        targets = targets.masked_fill(ignore, 0)
        true_distribution.scatter_(1, targets.unsqueeze(1), confidence)

    loss = torch.nn.functional.kl_div(
        log_probabilities, true_distribution, reduction="none"
    )
    loss = loss.masked_fill(ignore.unsqueeze(1), 0)

    # return loss according to reduction specified
    if reduction == "mean":
        return loss.sum().mean()
    elif reduction == "batchmean":
        return loss.sum() / bz
    elif reduction == "batch":
        return loss.view(bz, -1).sum(1) / length
    elif reduction == "sum":
        return loss.sum()
    else:
        return loss



def pesq_loss(y_true_batch, y_pred_batch, lens, reduction="mean"):
    """Compute the PESQ score and return -1 * that score.

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
    >>> -pesq_loss(b, a, torch.ones(1))
    tensor(0.7...)
    """
    def bark2hz(n, fs, n_fft):
        hz = []
        for k in np.linspace(0, 21, n+1):
            if k < 2:
                k = (k-0.3)/0.85
            elif k > 20.1:
                k = (k+4.422)/1.22
            h = 1960* (k+0.53) / (26.28-k)
            hz.append(int(2 * h / fs * (n_fft // 2 + 1)))
        return hz

    N = 49
    n_fft = 1024
    fs = 16000  # Sampling rate
    Sp = 6.910853e-1
    Sl = 1.866055e-1
    zwicker_power = 0.23
    D_POW_F = 2
    D_POW_S = 6
    D_POW_T = 2

    A_POW_F = 1
    A_POW_S = 6
    A_POW_T = 2

    D_WEIGHT = 0.1
    A_WEIGHT = 0.0309


    abs_thresh_power = torch.tensor([51286152.000000,     2454709.500000,     70794.593750,     4897.788574,     1174.897705,     
                                    389.045166,     104.712860,     45.708820,     17.782795,     9.772372,     
                                    4.897789,     3.090296,     1.905461,     1.258925,     0.977237,     
                                    0.724436,     0.562341,     0.457088,     0.389045,     0.331131,     
                                    0.295121,     0.269153,     0.257040,     0.251189,     0.251189,     
                                    0.251189,     0.251189,     0.263027,     0.288403,     0.309030,     
                                    0.338844,     0.371535,     0.398107,     0.436516,     0.467735,     
                                    0.489779,     0.501187,     0.501187,     0.512861,     0.524807,     
                                    0.524807,     0.524807,     0.512861,     0.478630,     0.426580,     
                                    0.371535,     0.363078,     0.416869,     0.537032])

    pow_dens_correction_factor = [100.000000,     99.999992,     100.000000,     100.000008,     100.000008,     
                                    100.000015,     99.999992,     99.999969,     50.000027,     100.000000,     
                                    99.999969,     100.000015,     99.999947,     100.000061,     53.047077,     
                                    110.000046,     117.991989,     65.000000,     68.760147,     69.999931,     
                                    71.428818,     75.000038,     76.843384,     80.968781,     88.646126,     
                                    63.864388,     68.155350,     72.547775,     75.584831,     58.379192,     
                                    80.950836,     64.135651,     54.384785,     73.821884,     64.437073,     
                                    59.176456,     65.521278,     61.399822,     58.144047,     57.004543,     
                                    64.126297,     54.311001,     61.114979,     55.077751,     56.849335,     
                                    55.628868,     53.137054,     54.985844,     79.546974]

    h = torch.tensor([2.00,    2.00,     2.00,     2.00,     1.82,     
                        1.66,   1.51,     1.39,     1.29,     1.20,     
                        1.12,   1.05,     1.00,     1.00,     1.00,     
                        1.00,   1.00,     1.00,     1.00,     1.00,     
                        1.00,   1.00,     1.00,     1.00,     1.00,      
                        1.00,   1.00,     1.00,     1.00,     1.00,     
                        1.00,   1.00,     1.00,     1.00,     1.00,      
                        1.00,   1.00,     1.00,     1.00,     1.00,      
                        1.00,   1.00,     1.00,     1.00,     1.00,      
                        1.00,   1.00,     1.00,     1.00])
    
    width_of_band_bark = torch.tensor([0.157344,     0.317994,     0.322441,     0.326934,     0.331474,     
                                        0.336061,     0.340697,     0.345381,     0.350114,     0.354897,     
                                        0.359729,     0.364611,     0.369544,     0.374529,     0.379565,     
                                        0.384653,     0.389794,     0.394989,     0.400236,     0.405538,     
                                        0.410894,     0.416306,     0.421773,     0.427297,     0.432877,     
                                        0.438514,     0.444209,     0.449962,     0.455774,     0.461645,     
                                        0.467577,     0.473569,     0.479621,     0.485736,     0.491912,     
                                        0.498151,     0.504454,     0.510819,     0.517250,     0.523745,     
                                        0.530308,     0.536934,     0.543629,     0.550390,     0.557220,     
                                        0.564119,     0.571085,     0.578125,     0.585232])


    barkscale = bark2hz(N, fs, n_fft)
    y_pred_batch = y_pred_batch.cpu()

    #[B, T]
    y_pred_batch = torch.squeeze(y_pred_batch, dim=-1)
    y_true_batch = torch.squeeze(y_true_batch, dim=-1)

    batch_size = y_pred_batch.shape[0]

    
    for i in range(0, batch_size):
        y_true = y_true_batch[i].cpu()
        y_pred = y_pred_batch[i].cpu()
        #[F, T]
        stft_true = torchaudio.transforms.Spectrogram(n_fft=n_fft, win_length=n_fft//2, hop_length=n_fft//4, power=2)(y_true)
        stft_pred = torchaudio.transforms.Spectrogram(n_fft=n_fft, win_length=n_fft//2, hop_length=n_fft//4, power=2)(y_pred)
        
        #Level Alignment
        low_f = int(2 * 300 / fs * (n_fft // 2 + 1))
        high_f = int(2 * 3000 / fs * (n_fft // 2 + 1))
        energy_true = torch.mean(stft_true[low_f:high_f, :]) + 1e-14
        energy_pred = torch.mean(stft_pred[low_f:high_f, :]) + 1e-14
        stft_true = stft_true * 1e7 / energy_true
        stft_pred = stft_pred * 1e7 / energy_pred

        #Bark Spectrum
        _, T = stft_true.shape
        B_true = torch.zeros((N, T))
        B_pred = torch.zeros((N, T))
        for j in range(len(barkscale)-1):
            B_true[j] = torch.sum(stft_true[barkscale[j] : barkscale[j+1]], dim=0) * pow_dens_correction_factor[j] * Sp
            B_pred[j] = torch.sum(stft_pred[barkscale[j] : barkscale[j+1]], dim=0) * pow_dens_correction_factor[j] * Sp
        threshold = abs_thresh_power.unsqueeze(1).repeat(1, T) * 1e4
        mask_true = torch.where(B_true > threshold, 1.0, 0.0)
        mask_pred = torch.where(B_pred > threshold, 1.0, 0.0)
        total_true = torch.sum(B_true * mask_true, dim=0)
        #print(total_true)
        #print(torch.sum(torch.where(total_true > 1e3, 1.0, 0.0)) / total_true.shape[0])
        nsilence = torch.where(total_true > 1e7, 1.0, 0.0)

        # #TF Equalization
        avg_B_true = torch.mean(B_true * mask_true * nsilence, dim=1, keepdim=True)
        avg_B_pred = torch.mean(B_pred * mask_pred * nsilence, dim=1, keepdim=True)
        # print(avg_B_true, avg_B_pred)
        scale = (avg_B_pred + 1e3) / (avg_B_true + 1e3)
        scale = torch.clamp(scale, 0.01, 100)
        B_true = B_true * scale
        mask_true = torch.where(B_true > threshold, 1.0, 0.0)
        mask_pred = torch.where(B_pred > threshold, 1.0, 0.0)
        total_true = torch.sum(B_true * mask_true, dim=0)
        total_pred = torch.sum(B_pred * mask_pred, dim=0)
        scale = torch.zeros_like(B_pred)
        s = 1.0
        for t in range(T):
            s = 0.2*s + (total_true[t] + 5e3) / (total_pred[t] + 5e3)
            scale[:, t] = torch.clamp(s, 3e-4, 5.0)
        B_pred = B_pred * scale

        #Loudness Mapping
        modified_zwicker_power = (h**0.15 * zwicker_power).unsqueeze(1).repeat(1, T)
        B_pred = (2 * threshold)**modified_zwicker_power * ((0.5 + 0.5 * B_pred / threshold)**modified_zwicker_power - 1) * mask_pred * Sl
        B_true = (2 * threshold)**modified_zwicker_power * ((0.5 + 0.5 * B_true / threshold)**modified_zwicker_power - 1) * mask_true * Sl
        #print(B_pred.mean(dim=1), B_true.mean(dim=1))

        #Disturbance Processing
        d = B_pred - B_true
        m = torch.where(B_pred < B_true, B_pred, B_true) * 0.25
        distance = torch.where(d > m, d - m, torch.zeros_like(d))
        distance += torch.where(d < -m, d + m, torch.zeros_like(d))
        #print(torch.mean(distance, dim=1))
        w = width_of_band_bark.unsqueeze(1).repeat(1, T)
        d = torch.abs(distance)
        sym = torch.sum((d*w)**D_POW_F, dim=0) / torch.sum(w, dim=0)
        sym = sym**(1.0 / D_POW_F) * torch.sum(w, dim=0)

        ratio = (B_pred + 50) / (B_true + 50)
        h = ratio ** 1.2
        h = torch.where(h < 3, torch.zeros_like(h), h)
        h = torch.clamp(h, 0, 12)
        distance = distance * h
        d = torch.abs(distance)
        asym = torch.sum((d*w)**A_POW_F, dim=0) / torch.sum(w, dim=0)
        asym = asym**(1.0 / A_POW_F) * torch.sum(w, dim=0)

        #Aggregation
        h = ((total_true + 1E5) / 1E7) ** 0.04
        sym = torch.clamp(sym / h, -float('inf'), 45)
        asym = torch.clamp(asym / h, -float('inf'), 45)

        NUMBER_OF_PSQM_FRAMES_PER_SYLLABE = 20
        #[N, K]
        sframe = F.unfold(sym.reshape(1,1,1,-1), kernel_size=(1,NUMBER_OF_PSQM_FRAMES_PER_SYLLABE), 
                    stride=(1,NUMBER_OF_PSQM_FRAMES_PER_SYLLABE // 2)).squeeze(0)
        asframe = F.unfold(asym.reshape(1,1,1,-1), kernel_size=(1,NUMBER_OF_PSQM_FRAMES_PER_SYLLABE), 
                    stride=(1,NUMBER_OF_PSQM_FRAMES_PER_SYLLABE // 2)).squeeze(0)
        left = len(sym) - sframe.shape[0] * sframe.shape[1] // 2
        sframe = torch.cat([torch.mean(sframe**D_POW_S, dim=0), torch.mean(sym[-left:]**D_POW_S).unsqueeze(0)]) + 1e-8
        asframe = torch.cat([torch.mean(asframe**A_POW_S, dim=0), torch.mean(asym[-left:]**A_POW_S).unsqueeze(0)]) + 1e-8
        sres = (torch.mean((sframe ** (1.0 / D_POW_S)) ** D_POW_T) + 1e-8) ** (1.0 / D_POW_T)
        asres = (torch.mean((asframe ** (1.0 / A_POW_S)) ** A_POW_T) + 1e-8) ** (1.0 / A_POW_T)
        res = 4.5 - D_WEIGHT * sres - A_WEIGHT * asres
    return -res


    



def stoi_loss(y_true_batch, y_pred_batch, lens, reduction="mean"):
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
        if y_sil_true.shape[-1] <= 512:
            D[i] = 0.99
            continue
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



def yin(y_frames, fmin, fmax, sr=16000, frame_length=3200, win_length=400, hop_length=160, trough_threshold=0.1):
    def _cumulative_mean_normalized_difference(y_frames, frame_length, win_length, min_period, max_period):
        # Autocorrelation.
        a = np.fft.rfft(y_frames, frame_length, axis=0)
        b = np.fft.rfft(y_frames[win_length::-1, :], frame_length, axis=0)
        acf_frames = np.fft.irfft(a * b, frame_length, axis=0)[win_length:]
        acf_frames[np.abs(acf_frames) < 1e-6] = 0

        # Energy terms.
        energy_frames = np.cumsum(y_frames ** 2, axis=0)
        energy_frames = energy_frames[win_length:, :] - energy_frames[:-win_length, :]
        energy_frames[np.abs(energy_frames) < 1e-6] = 0

        # Difference function.
        yin_frames = energy_frames[0, :] + energy_frames - 2 * acf_frames

        # Cumulative mean normalized difference function.
        yin_numerator = yin_frames[min_period : max_period + 1, :]
        tau_range = np.arange(1, max_period + 1)[:, None]
        cumulative_mean = np.cumsum(yin_frames[1 : max_period + 1, :], axis=0) / tau_range
        yin_denominator = cumulative_mean[min_period - 1 : max_period, :]
        yin_frames = yin_numerator / (yin_denominator + EPS)
        return yin_frames
    
    def _parabolic_interpolation(y_frames):
        parabolic_shifts = np.zeros_like(y_frames)
        parabola_a = (y_frames[:-2, :] + y_frames[2:, :] - 2 * y_frames[1:-1, :]) / 2
        parabola_b = (y_frames[2:, :] - y_frames[:-2, :]) / 2
        parabolic_shifts[1:-1, :] = -parabola_b / (2 * parabola_a + EPS)
        parabolic_shifts[np.abs(parabolic_shifts) > 1] = 0
        return parabolic_shifts

    def _localmin(x, axis=0):
        paddings = [(0, 0)] * x.ndim
        paddings[axis] = (1, 1)

        x_pad = np.pad(x, paddings, mode="edge")

        inds1 = [slice(None)] * x.ndim
        inds1[axis] = slice(0, -2)

        inds2 = [slice(None)] * x.ndim
        inds2[axis] = slice(2, x_pad.shape[axis])

        return (x < x_pad[tuple(inds1)]) & (x <= x_pad[tuple(inds2)])


    # Set the default window length if it is not already specified.
    if win_length is None:
        win_length = frame_length // 2

    # Set the default hop if it is not already specified.
    if hop_length is None:
        hop_length = frame_length // 4


    # Calculate minimum and maximum periods
    min_period = max(int(np.floor(sr / fmax)), 1)
    max_period = min(int(np.ceil(sr / fmin)), frame_length - win_length - 1)

    # Calculate cumulative mean normalized difference function.
    yin_frames = _cumulative_mean_normalized_difference(
        y_frames, frame_length, win_length, min_period, max_period
    )
    # Parabolic interpolation.
    parabolic_shifts = _parabolic_interpolation(yin_frames)

    # Find local minima.
    is_trough = _localmin(yin_frames, axis=0)
    is_trough[0, :] = yin_frames[0, :] < yin_frames[1, :]

    # Find minima below peak threshold.
    is_threshold_trough = np.logical_and(is_trough, yin_frames < trough_threshold)
    # Absolute threshold.
    # "The solution we propose is to set an absolute threshold and choose the
    # smallest value of tau that gives a minimum of d' deeper than
    # this threshold. If none is found, the global minimum is chosen instead."
    global_min = np.argmin(yin_frames, axis=0)
    yin_period = np.argmax(is_threshold_trough, axis=0)
    no_trough_below_threshold = np.all(~is_threshold_trough, axis=0)
    yin_period[no_trough_below_threshold] = global_min[no_trough_below_threshold]

    # Refine peak by parabolic interpolation.
    yin_period = (
        min_period
        + yin_period
        + parabolic_shifts[yin_period, range(yin_frames.shape[1])]
    )

    # Convert period to fundamental frequency.
    f0 = sr / yin_period
    return f0


if __name__ == '__main__':
    import soundfile as sf
    import librosa
    import librosa.display
    import matplotlib.pyplot as plt

    def plot_melspectrogram(wav, sr, n_mels=80, fmax=8000):
        fig, ax = plt.subplots(1,1,figsize=(20,5))
        M = librosa.feature.melspectrogram(y=wav, sr=sr, n_mels=n_mels, fmax=fmax, n_fft=400)
        M_db = librosa.power_to_db(M, ref=np.max)
        img = librosa.display.specshow(M_db, sr=sr, y_axis='mel', x_axis='time', ax=ax, fmax=fmax)
        ax.set(title='Mel spectrogram display')
        fig.colorbar(img, ax=ax, format="%+2.f dB")
        plt.savefig('./tmp.png')

    wav = sf.read('/nas/datasets/Chinese_Speech_Datasets/MAGICDATA/test/38_5752/38_5752_20170915222554.wav')[0]
    frames, pad = segmentation(torch.tensor(wav)[None, None, :], 3200)
    frames = frames.squeeze(1).T.numpy()
    # f0 = yin(frames, fmin=65, fmax=2093)
    # print(f0)
    # stft = torch.stft(torch.tensor(frames.T), n_fft=400, hop_length=160, return_complex=True).abs().numpy()
    # energy = 20*np.log10(np.mean(stft,axis=(1))+EPS)-10*np.log10(1e-8)
    # print((energy-np.min(energy))/(np.max(energy)-np.min(energy)))
    # print(np.mean(f0), np.mean(energy))
    # print(np.std(f0), np.std(energy))
    # print(np.where(energy<0.6))
    # print(np.where((f0<70) | (f0>1000)))
    stft = torch.stft(torch.tensor(frames.T), n_fft=400, hop_length=160, return_complex=True).abs().numpy()
    prob = np.clip(stft/(np.sum(stft**2, axis=1, keepdims=True)+1e-8),1e-8,1)
    H = -np.sum(prob*np.log(prob), axis=1)
    print(np.mean(H,axis=-1))
    print(np.where(H > 16))
    plot_melspectrogram(wav, 16000)

