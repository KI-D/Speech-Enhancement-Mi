
import math
import torch
import torchaudio
from speechbrain.dataio.dataio import read_audio
from speechbrain.processing.signal_processing import compute_amplitude, dB_to_amplitude, convolve1d, notch_filter, reverberate

torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.deterministic = True

class AddNoise(torch.nn.Module):
    '''
    from augment import AddNoise
    import pandas as pd
    csv_file = pd.read_csv('./data/test_noise.csv')
    signal = read_audio('./samples/sample_wave.wav')
    addnoise = AddNoise(csv_file)
    noisy = addnoise(signal)
    '''
    def __init__(self, csv_file = None, mix_prob = 1.0, snr_low = -5, snr_high = 5):
        super().__init__()
        self.csv_file = csv_file
        self.mix_prob = mix_prob
        self.snr_low = snr_low
        self.snr_high = snr_high
        

    def forward(self, waveform, rir_func=None, RIR=None):
        """
        Arguments
        ---------
        waveforms : tensor
            Shape should be `[time]` or `[time, channels]`.

        Returns
        -------
        Tensor of shape `[time]` or `[time, channels]`.
        """
        noisy_waveform = waveform.clone()
        length = len(waveform)

        # Don't add noise (return early) 1-`mix_prob` portion of the batches
        if torch.rand(1) > self.mix_prob:
            return noisy_waveform, torch.zeros_like(waveform)

        # Compute the average amplitude of the clean waveforms
        clean_amplitude = compute_amplitude(waveform.unsqueeze(0)).squeeze()

        # Pick an SNR and use it to compute the mixture amplitude factors
        SNR = torch.rand(1, device=waveform.device)
        SNR = SNR * (self.snr_high - self.snr_low) + self.snr_low
        noise_amplitude_factor = 1 / (dB_to_amplitude(SNR).squeeze() + 1)
        new_noise_amplitude = noise_amplitude_factor * clean_amplitude

        # Scale clean signal appropriately
        noisy_waveform *= 1 - noise_amplitude_factor

        # Loop through clean samples and create mixture
        if self.csv_file is None:
            white_noise = torch.randn_like(waveform)
            noise = new_noise_amplitude * white_noise
        else:
            noise = self.load_noise(length, rir_func, RIR)
            if noise.dim() != noisy_waveform.dim():
                noise = torch.stack([noise]*3, dim=1)
            noise_amplitude = compute_amplitude(noise.unsqueeze(0)).squeeze()
            if noisy_waveform.dim() >= 2 and noise.dim() < 2:
                noise = noise.unsqueeze(dim=-1)
            # Rescale and add
            noise *= new_noise_amplitude / (noise_amplitude + 1e-8)

        noisy_waveform += noise
        # Normalizing to prevent clipping
        abs_max, _ = torch.max(torch.abs(noisy_waveform), dim=0, keepdim=True)
        noisy_waveform = noisy_waveform / abs_max.clamp(min=1.0)
        return noisy_waveform, noise
    
    def load_noise(self, length, rir_func=None, RIR=None):
        N_noise = len(self.csv_file)
        ind = int(torch.randint(N_noise, (1,)))
        if 'origin_path' in self.csv_file.columns:
            name = 'origin_path'
        else:
            name = 'path'
        noise, sr = torchaudio.load(self.csv_file.loc[ind, name])
        if noise.dim() > 1:
            noise = noise[0]
        if rir_func is not None:
            noise = rir_func.simulate(sources=noise, RIR=RIR).transpose(1,0)
            #noise,_ = rir_func.simulate(sources=[noise], param=RIR)
            #noise = noise[0].transpose(1,0)
        noise_len = len(noise)
        if length > noise_len:
            noise, noise_len = self.extend_noise(noise, length, sr)
        if noise_len > length:
            chop = noise_len - length
            start_index = torch.randint(high=chop, size=(1,))
            # Truncate noise_batch to max_length
            noise = noise[start_index : start_index + length]
        return noise
    
    def extend_noise(self, noise, max_length, sr):
        """ Concatenate noise using hanning window"""
        window = torch.hann_window(sr + 1)
        if len(noise) < sr+1:
            gs_noise = torch.randn_like(window)
            if noise.dim()>1:
                 gs_noise = gs_noise.unsqueeze(1).repeat(1,noise.shape[1])
            gs_noise[:len(noise)] += noise
            noise = gs_noise
        
        noise_ex = noise
        # Increasing window
        i_w = window[:(len(window) // 2 + 1)]
        # Decreasing window
        d_w = i_w.flip(0)
        if noise.dim()>1:
            i_w = i_w.unsqueeze(1)
            d_w = d_w.unsqueeze(1)
        iw_len = len(i_w)
        dw_len = len(d_w)
        noise_len = len(noise_ex)
        # Extend until length is reached
        while noise_len < max_length:
            noise_ex = torch.cat((noise_ex[:(noise_len-dw_len)], noise_ex[(noise_len-dw_len):]*d_w + noise[:iw_len]*i_w, noise[iw_len:]))
            noise_len = len(noise_ex)
        return noise_ex, noise_len


class AddReverb(torch.nn.Module):
    '''
    from augment import AddReverb
    import pandas as pd
    csv_file = pd.read_csv('./data/test_rir.csv')
    signal = read_audio('./samples/sample_wave.wav')
    addreverb = AddReverb(csv_file)

    noisy = addreverb(signal)
    '''
    def __init__(self, csv_file, reverb_prob=1.0, rir_scale_factor=1.0):
        super().__init__()
        self.csv_file = csv_file
        self.reverb_prob = reverb_prob
        self.rir_scale_factor = rir_scale_factor

    def forward(self, waveforms):
        """
        Arguments
        ---------
        waveforms : tensor
            Shape should be `[time]` or `[time, channels]`.

        Returns
        -------
        Tensor of shape `[time]` or `[time, channels]`.
        """

        # Don't add reverb (return early) 1-`reverb_prob` portion of the time
        if torch.rand(1) > self.reverb_prob:
            return waveforms.clone()

        # Load and prepare RIR
        rir_waveform = self.load_rir()
        # Compress or dilate RIR
        if self.rir_scale_factor != 1:
            rir_waveform = F.interpolate(rir_waveform.transpose(1, -1), scale_factor=self.rir_scale_factor, 
                                        mode="linear", align_corners=False)
            rir_waveform = rir_waveform.transpose(1, -1)
        rev_waveform = reverberate(waveforms, rir_waveform, rescale_amp="avg").squeeze()
        return rev_waveform

    def load_rir(self):
        N_rir = len(self.csv_file)
        ind = int(torch.randint(N_rir, (1,)))
        rir, sr = torchaudio.load(self.csv_file.loc[ind, 'path'])
        rir = rir.transpose(0, 1).squeeze()
        if rir.dim() > 1:
            rir = rir[:,0]
        return rir


class SpeedPerturb(torch.nn.Module):
    """
    from augment import SpeedPerturb
    signal = read_audio('./samples/sample_wave.wav')
    perturbator = SpeedPerturb(orig_freq=16000)
    perturbed = perturbator(signal)
    """

    def __init__(self, orig_freq, speeds=[90, 95, 105, 110], perturb_prob=1.0):
        super().__init__()
        self.orig_freq = orig_freq
        self.speeds = speeds
        self.perturb_prob = perturb_prob

        # Initialize index of perturbation
        self.samp_index = 0

        # Initialize resamplers
        self.resamplers = []
        for speed in self.speeds:
            config = {"orig_freq": self.orig_freq, "new_freq": self.orig_freq * speed // 100}
            self.resamplers.append(Resample(**config))

    def forward(self, waveform):
        """
        Arguments
        ---------
        waveforms : tensor
            Shape should be `[time]` or `[time, channels]`.

        Returns
        -------
        Tensor of shape `[time]` or `[time, channels]`.
        """

        # Don't perturb (return early) 1-`perturb_prob` portion of the batches
        if torch.rand(1) > self.perturb_prob:
            return waveform.clone()
        # Add channels dimension
        if len(waveform.shape) == 2:
            waveform = waveform.unsqueeze(0)
        else:
            waveform = waveform.unsqueeze(0).unsqueeze(-1)
        waveform = waveform.to(torch.float32)

        # Perform a random perturbation
        samp_index = torch.randint(len(self.speeds), (1,))[0]
        perturbed_waveform = self.resamplers[samp_index](waveform)
        return perturbed_waveform.squeeze()


class Resample(torch.nn.Module):
    """This class resamples an audio signal using sinc-based interpolation.

    It is a modification of the `resample` function from torchaudio
    (https://pytorch.org/audio/transforms.html#resample)

    Arguments
    ---------
    orig_freq : int
        the sampling frequency of the input signal.
    new_freq : int
        the new sampling frequency after this operation is performed.
    lowpass_filter_width : int
        Controls the sharpness of the filter, larger numbers result in a
        sharper filter, but they are less efficient. Values from 4 to 10 are
        allowed.

    Example
    -------
    >>> from speechbrain.dataio.dataio import read_audio
    >>> signal = read_audio('samples/audio_samples/example1.wav')
    >>> signal = signal.unsqueeze(0) # [batch, time, channels]
    >>> resampler = Resample(orig_freq=16000, new_freq=8000)
    >>> resampled = resampler(signal)
    >>> signal.shape
    torch.Size([1, 52173])
    >>> resampled.shape
    torch.Size([1, 26087])
    """

    def __init__(
        self, orig_freq=16000, new_freq=16000, lowpass_filter_width=6,
    ):
        super().__init__()
        self.orig_freq = orig_freq
        self.new_freq = new_freq
        self.lowpass_filter_width = lowpass_filter_width

        # Compute rate for striding
        self._compute_strides()
        assert self.orig_freq % self.conv_stride == 0
        assert self.new_freq % self.conv_transpose_stride == 0

    def _compute_strides(self):
        """Compute the phases in polyphase filter.

        (almost directly from torchaudio.compliance.kaldi)
        """

        # Compute new unit based on ratio of in/out frequencies
        base_freq = math.gcd(self.orig_freq, self.new_freq)
        input_samples_in_unit = self.orig_freq // base_freq
        self.output_samples = self.new_freq // base_freq

        # Store the appropriate stride based on the new units
        self.conv_stride = input_samples_in_unit
        self.conv_transpose_stride = self.output_samples

    def forward(self, waveforms):
        """
        Arguments
        ---------
        waveforms : tensor
            Shape should be `[batch, time]` or `[batch, time, channels]`.
        lengths : tensor
            Shape should be a single dimension, `[batch]`.

        Returns
        -------
        Tensor of shape `[batch, time]` or `[batch, time, channels]`.
        """

        if not hasattr(self, "first_indices"):
            self._indices_and_weights(waveforms)

        # Don't do anything if the frequencies are the same
        if self.orig_freq == self.new_freq:
            return waveforms

        unsqueezed = False
        if len(waveforms.shape) == 2:
            waveforms = waveforms.unsqueeze(1)
            unsqueezed = True
        elif len(waveforms.shape) == 3:
            waveforms = waveforms.transpose(1, 2)
        else:
            raise ValueError("Input must be 2 or 3 dimensions")

        # Do resampling
        resampled_waveform = self._perform_resample(waveforms)

        if unsqueezed:
            resampled_waveform = resampled_waveform.squeeze(1)
        else:
            resampled_waveform = resampled_waveform.transpose(1, 2)

        return resampled_waveform

    def _perform_resample(self, waveforms):
        """Resamples the waveform at the new frequency.

        This matches Kaldi's OfflineFeatureTpl ResampleWaveform which uses a
        LinearResample (resample a signal at linearly spaced intervals to
        up/downsample a signal). LinearResample (LR) means that the output
        signal is at linearly spaced intervals (i.e the output signal has a
        frequency of `new_freq`). It uses sinc/bandlimited interpolation to
        upsample/downsample the signal.

        (almost directly from torchaudio.compliance.kaldi)

        https://ccrma.stanford.edu/~jos/resample/
        Theory_Ideal_Bandlimited_Interpolation.html

        https://github.com/kaldi-asr/kaldi/blob/master/src/feat/resample.h#L56

        Arguments
        ---------
        waveforms : tensor
            The batch of audio signals to resample.

        Returns
        -------
        The waveforms at the new frequency.
        """

        # Compute output size and initialize
        batch_size, num_channels, wave_len = waveforms.size()
        window_size = self.weights.size(1)
        tot_output_samp = self._output_samples(wave_len)
        resampled_waveform = torch.zeros(
            (batch_size, num_channels, tot_output_samp),
            device=waveforms.device,
        )
        self.weights = self.weights.to(waveforms.device)

        # Check weights are on correct device
        if waveforms.device != self.weights.device:
            self.weights = self.weights.to(waveforms.device)

        # eye size: (num_channels, num_channels, 1)
        eye = torch.eye(num_channels, device=waveforms.device).unsqueeze(2)

        # Iterate over the phases in the polyphase filter
        for i in range(self.first_indices.size(0)):
            wave_to_conv = waveforms
            first_index = int(self.first_indices[i].item())
            if first_index >= 0:
                # trim the signal as the filter will not be applied
                # before the first_index
                wave_to_conv = wave_to_conv[..., first_index:]

            # pad the right of the signal to allow partial convolutions
            # meaning compute values for partial windows (e.g. end of the
            # window is outside the signal length)
            max_index = (tot_output_samp - 1) // self.output_samples
            end_index = max_index * self.conv_stride + window_size
            current_wave_len = wave_len - first_index
            right_padding = max(0, end_index + 1 - current_wave_len)
            left_padding = max(0, -first_index)
            wave_to_conv = torch.nn.functional.pad(
                wave_to_conv, (left_padding, right_padding)
            )
            conv_wave = torch.nn.functional.conv1d(
                input=wave_to_conv,
                weight=self.weights[i].repeat(num_channels, 1, 1),
                stride=self.conv_stride,
                groups=num_channels,
            )

            # we want conv_wave[:, i] to be at
            # output[:, i + n*conv_transpose_stride]
            dilated_conv_wave = torch.nn.functional.conv_transpose1d(
                conv_wave, eye, stride=self.conv_transpose_stride
            )

            # pad dilated_conv_wave so it reaches the output length if needed.
            left_padding = i
            previous_padding = left_padding + dilated_conv_wave.size(-1)
            right_padding = max(0, tot_output_samp - previous_padding)
            dilated_conv_wave = torch.nn.functional.pad(
                dilated_conv_wave, (left_padding, right_padding)
            )
            dilated_conv_wave = dilated_conv_wave[..., :tot_output_samp]

            resampled_waveform += dilated_conv_wave

        return resampled_waveform

    def _output_samples(self, input_num_samp):
        """Based on LinearResample::GetNumOutputSamples.

        LinearResample (LR) means that the output signal is at
        linearly spaced intervals (i.e the output signal has a
        frequency of ``new_freq``). It uses sinc/bandlimited
        interpolation to upsample/downsample the signal.

        (almost directly from torchaudio.compliance.kaldi)

        Arguments
        ---------
        input_num_samp : int
            The number of samples in each example in the batch.

        Returns
        -------
        Number of samples in the output waveform.
        """

        # For exact computation, we measure time in "ticks" of 1.0 / tick_freq,
        # where tick_freq is the least common multiple of samp_in and
        # samp_out.
        samp_in = int(self.orig_freq)
        samp_out = int(self.new_freq)

        tick_freq = abs(samp_in * samp_out) // math.gcd(samp_in, samp_out)
        ticks_per_input_period = tick_freq // samp_in

        # work out the number of ticks in the time interval
        # [ 0, input_num_samp/samp_in ).
        interval_length = input_num_samp * ticks_per_input_period
        if interval_length <= 0:
            return 0
        ticks_per_output_period = tick_freq // samp_out

        # Get the last output-sample in the closed interval,
        # i.e. replacing [ ) with [ ]. Note: integer division rounds down.
        # See http://en.wikipedia.org/wiki/Interval_(mathematics) for an
        # explanation of the notation.
        last_output_samp = interval_length // ticks_per_output_period

        # We need the last output-sample in the open interval, so if it
        # takes us to the end of the interval exactly, subtract one.
        if last_output_samp * ticks_per_output_period == interval_length:
            last_output_samp -= 1

        # First output-sample index is zero, so the number of output samples
        # is the last output-sample plus one.
        num_output_samp = last_output_samp + 1

        return num_output_samp

    def _indices_and_weights(self, waveforms):
        """Based on LinearResample::SetIndexesAndWeights

        Retrieves the weights for resampling as well as the indices in which
        they are valid. LinearResample (LR) means that the output signal is at
        linearly spaced intervals (i.e the output signal has a frequency
        of ``new_freq``). It uses sinc/bandlimited interpolation to
        upsample/downsample the signal.

        Returns
        -------
        - the place where each filter should start being applied
        - the filters to be applied to the signal for resampling
        """

        # Lowpass filter frequency depends on smaller of two frequencies
        min_freq = min(self.orig_freq, self.new_freq)
        lowpass_cutoff = 0.99 * 0.5 * min_freq

        assert lowpass_cutoff * 2 <= min_freq
        window_width = self.lowpass_filter_width / (2.0 * lowpass_cutoff)

        assert lowpass_cutoff < min(self.orig_freq, self.new_freq) / 2
        output_t = torch.arange(
            start=0.0, end=self.output_samples, device=waveforms.device,
        )
        output_t /= self.new_freq
        min_t = output_t - window_width
        max_t = output_t + window_width

        min_input_index = torch.ceil(min_t * self.orig_freq)
        max_input_index = torch.floor(max_t * self.orig_freq)
        num_indices = max_input_index - min_input_index + 1

        max_weight_width = num_indices.max()
        j = torch.arange(max_weight_width, device=waveforms.device)
        input_index = min_input_index.unsqueeze(1) + j.unsqueeze(0)
        delta_t = (input_index / self.orig_freq) - output_t.unsqueeze(1)

        weights = torch.zeros_like(delta_t)
        inside_window_indices = delta_t.abs().lt(window_width)

        # raised-cosine (Hanning) window with width `window_width`
        weights[inside_window_indices] = 0.5 * (
            1
            + torch.cos(
                2
                * math.pi
                * lowpass_cutoff
                / self.lowpass_filter_width
                * delta_t[inside_window_indices]
            )
        )

        t_eq_zero_indices = delta_t.eq(0.0)
        t_not_eq_zero_indices = ~t_eq_zero_indices

        # sinc filter function
        weights[t_not_eq_zero_indices] *= torch.sin(
            2 * math.pi * lowpass_cutoff * delta_t[t_not_eq_zero_indices]
        ) / (math.pi * delta_t[t_not_eq_zero_indices])

        # limit of the function at t = 0
        weights[t_eq_zero_indices] *= 2 * lowpass_cutoff

        # size (output_samples, max_weight_width)
        weights /= self.orig_freq

        self.first_indices = min_input_index
        self.weights = weights


class DropFreq(torch.nn.Module):
    """
    from speechbrain.dataio.dataio import read_audio
    dropper = DropFreq()
    signal = read_audio('./samples/sample_wave.wav')
    dropped_signal = dropper(signal)
    """

    def __init__(self, drop_freq_low=1e-14, drop_freq_high=1, drop_count_low=1, drop_count_high=2, drop_width=0.05, drop_prob=1):
        super().__init__()
        self.drop_freq_low = drop_freq_low
        self.drop_freq_high = drop_freq_high
        self.drop_count_low = drop_count_low
        self.drop_count_high = drop_count_high
        self.drop_width = drop_width
        self.drop_prob = drop_prob

    def forward(self, waveforms):
        """
        Arguments
        ---------
        waveforms : tensor
            Shape should be `[time]` or `[time, channels]`.

        Returns
        -------
        Tensor of shape `[time]` or `[time, channels]`.
        """

        # Don't drop (return early) 1-`drop_prob` portion of the batches
        dropped_waveform = waveforms.clone().to(torch.float32)
        if torch.rand(1) > self.drop_prob:
            return dropped_waveform

        # Add channels dimension
        if len(waveforms.shape) == 2:
            dropped_waveform = dropped_waveform.unsqueeze(0)
        else:
            dropped_waveform = dropped_waveform.unsqueeze(0).unsqueeze(-1)

        # Pick number of frequencies to drop
        drop_count = torch.randint(low=self.drop_count_low, high=self.drop_count_high + 1, size=(1,))

        # Pick a frequency to drop
        drop_range = self.drop_freq_high - self.drop_freq_low
        drop_frequency = torch.rand(drop_count) * drop_range + self.drop_freq_low

        # Filter parameters
        filter_length = 101
        pad = filter_length // 2

        # Start with delta function
        drop_filter = torch.zeros(1, filter_length, 1)
        drop_filter[0, pad, 0] = 1

        # Subtract each frequency
        for frequency in drop_frequency:
            notch_kernel = notch_filter(frequency, filter_length, self.drop_width)
            drop_filter = convolve1d(drop_filter, notch_kernel, pad)

        # Apply filter
        dropped_waveform = convolve1d(dropped_waveform, drop_filter, pad)

        # Remove channels dimension if added
        return dropped_waveform.squeeze()


class DropChunk(torch.nn.Module):
    """
    from speechbrain.dataio.dataio import read_audio
    dropper = DropChunk(drop_start=100, drop_end=200, noise_factor=0.)
    signal = read_audio('samples/audio_samples/example1.wav')
    signal = signal.unsqueeze(0) # [batch, time, channels]
    dropped_signal = dropper(signal, length)
    """

    def __init__(self, drop_length_low=100, drop_length_high=500, drop_count_low_rate=0.00002, drop_count_high_rate=0.00008, drop_start=0, drop_end=None, drop_prob=1, noise_factor=0.0):
        super().__init__()
        self.drop_length_low = drop_length_low
        self.drop_length_high = drop_length_high
        self.drop_count_low_rate = drop_count_low_rate
        self.drop_count_high_rate = drop_count_high_rate
        self.drop_start = drop_start
        self.drop_end = drop_end
        self.drop_prob = drop_prob
        self.noise_factor = noise_factor

        # Validate low < high
        if drop_length_low > drop_length_high:
            raise ValueError("Low limit must not be more than high limit")

        # Make sure the length doesn't exceed end - start
        if drop_end is not None and drop_end >= 0:
            if drop_start > drop_end:
                raise ValueError("Low limit must not be more than high limit")

            drop_range = drop_end - drop_start
            self.drop_length_low = min(drop_length_low, drop_range)
            self.drop_length_high = min(drop_length_high, drop_range)

    def forward(self, waveforms):
        """
        Arguments
        ---------
        waveforms : tensor
            Shape should be `[time]` or `[time, channels]`.

        Returns
        -------
        Tensor of shape `[time]` or `[time, channels]`
        """

        # Reading input list
        dropped_waveform = waveforms.clone()

        # Don't drop (return early) 1-`drop_prob` portion of the batches
        if torch.rand(1) > self.drop_prob:
            return dropped_waveform
        
        wave_length = len(waveforms)
        # Add channels dimension
        if len(waveforms.shape) == 2:
            dropped_waveform = dropped_waveform.unsqueeze(0)
        else:
            dropped_waveform = dropped_waveform.unsqueeze(0).unsqueeze(-1)

        # Store original amplitude for computing white noise amplitude
        clean_amplitude = compute_amplitude(waveforms)

        self.drop_count_high = int(self.drop_count_high_rate * wave_length)
        self.drop_count_low = int(self.drop_count_low_rate * wave_length)
        self.drop_count_low = max(1, self.drop_count_low)
        self.drop_count_high = max(self.drop_count_low, self.drop_count_high)
        # Pick a number of times to drop
        drop_times = torch.randint(
            low=self.drop_count_low,
            high=self.drop_count_high + 1,
            size=(1,),
        )

        # Pick lengths
        length = torch.randint(
            low=self.drop_length_low,
            high=self.drop_length_high + 1,
            size=(drop_times[0],),
        )

        # Compute range of starting locations
        start_min = self.drop_start
        if start_min < 0:
            start_min += wave_length
        start_max = self.drop_end
        if start_max is None:
            start_max = wave_length
        if start_max < 0:
            start_max += wave_length
        start_max = start_max - length.max()

        # Pick starting locations
        start = torch.randint(
            low=start_min, high=start_max + 1, size=(drop_times[0],),
        )

        end = start + length

        # Update waveform
        if not self.noise_factor:
            for j in range(drop_times[0]):
                dropped_waveform[0, start[j] : end[j]] = 0.0
        else:
            # Uniform distribution of -2 to +2 * avg amplitude should
            # preserve the average for normalization
            noise_max = 2 * clean_amplitude[0] * self.noise_factor
            for j in range(drop_times[0]):
                # zero-center the noise distribution
                noise_vec = torch.rand(wave_length, device=waveforms.device)
                noise_vec = 2 * noise_max * noise_vec - noise_max
                dropped_waveform[0, start[j] : end[j]] = noise_vec

        return dropped_waveform.squeeze()


class DoClip(torch.nn.Module):
    """This function mimics audio clipping by clamping the input tensor.

    Arguments
    ---------
    clip_low : float
        The low end of amplitudes for which to clip the signal.
    clip_high : float
        The high end of amplitudes for which to clip the signal.
    clip_prob : float
        The probability that the batch of signals will have a portion clipped.
        By default, every batch has portions clipped.

    Example
    -------
    >>> from speechbrain.dataio.dataio import read_audio
    >>> clipper = DoClip(clip_low=0.01, clip_high=0.01)
    >>> signal = read_audio('samples/audio_samples/example1.wav')
    >>> clipped_signal = clipper(signal.unsqueeze(0))
    >>> "%.2f" % clipped_signal.max()
    '0.01'
    """

    def __init__(self, clip_low=0.5, clip_high=1, clip_prob=1):
        super().__init__()
        self.clip_low = clip_low
        self.clip_high = clip_high
        self.clip_prob = clip_prob

    def forward(self, waveforms):
        """
        Arguments
        ---------
        waveforms : tensor
            Shape should be `[time]` or `[time, channels]`.

        Returns
        -------
        Tensor of shape `[time]` or `[time, channels]`
        """

        # Don't clip (return early) 1-`clip_prob` portion of the batches
        if torch.rand(1) > self.clip_prob:
            return waveforms.clone()

        # Add channels dimension
        if len(waveforms.shape) == 2:
            waveforms= waveforms.unsqueeze(0)
        else:
            waveforms = waveforms.unsqueeze(0).unsqueeze(-1)

        # Randomly select clip value
        clipping_range = self.clip_high - self.clip_low
        clip_value = torch.rand(1,)[0] * clipping_range + self.clip_low

        # Apply clipping
        clipped_waveform = waveforms.clamp(-clip_value, clip_value)

        return clipped_waveform.squeeze()
