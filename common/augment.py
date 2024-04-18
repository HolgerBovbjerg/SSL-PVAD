from typing import Optional, Union

import torch
from torch_audiomentations import (AddBackgroundNoise, ApplyImpulseResponse,
                                   Compose)
from torchaudio.transforms import FrequencyMasking, TimeMasking, TimeStretch


class SpecAugment(torch.nn.Module):
    def __init__(self, time_mask_max_length: int = 30, p_max: float = 1., n_time_mask: int = 10,
                 freq_mask_max_length: int = 50, n_freq_mask: int = 2,
                 use_time_stretch: bool = False, rate_min: Optional[float] = 0.9, rate_max: Optional[float] = 1.2,
                 freq_bins: Optional[int] = 512):
        super().__init__()
        if use_time_stretch:
            self.stretch = TimeStretch(n_freq=freq_bins)
            self.rate_min = rate_min
            self.rate_max = rate_max
        self.time_mask = TimeMasking(time_mask_param=time_mask_max_length, p=p_max)
        self.n_time_mask = n_time_mask
        self.n_freq_mask = n_freq_mask
        self.freq_mask = FrequencyMasking(freq_mask_param=freq_mask_max_length)

    def forward(self, x):
        x = x.transpose(-1, -2).unsqueeze(0)
        # TODO: Make time stretch work for SpecAugment
        # x = self.stretch(x, overriding_rate=random.uniform(self.rate_min, self.rate_max))
        for _ in range(self.n_time_mask):
            x = self.time_mask(x)
        for _ in range(self.n_freq_mask):
            x = self.freq_mask(x)
        x = x.transpose(-1, -2).squeeze()
        return x


class AddRIR(torch.nn.Module):
    def __init__(self, rir_paths: Union[str, list], sampling_rate: int = 16000, p=1., convolve_mode="full",
                 compensate_for_propagation_delay=False, mode="per_example"):
        super().__init__()
        self.rir_paths = rir_paths
        self.sampling_rate = sampling_rate
        self.p = p
        self.convolve_mode = convolve_mode
        self.compensate_for_propagation_delay = compensate_for_propagation_delay
        self.mode = mode
        self.apply_rir = ApplyImpulseResponse(ir_paths=self.rir_paths,
                                              convolve_mode=self.convolve_mode,
                                              compensate_for_propagation_delay=self.compensate_for_propagation_delay,
                                              mode=self.mode,
                                              p=self.p,
                                              sample_rate=self.sampling_rate)

    def forward(self, x):
        return self.apply_rir(x)


class AddNoise(torch.nn.Module):
    def __init__(self, noise_paths: Union[str, list], sampling_rate: int = 16000, snr_db_min: int = 3,
                 snr_db_max: int = 30, p: float = 1.):
        super().__init__()
        self.noise_paths = noise_paths
        self.sampling_rate = sampling_rate
        self.snr_db_min = snr_db_min
        self.snr_db_max = snr_db_max
        self.p = p
        self.add_noise = AddBackgroundNoise(background_paths=noise_paths,
                                            sample_rate=self.sampling_rate,
                                            min_snr_in_db=self.snr_db_min,
                                            max_snr_in_db=self.snr_db_max,
                                            p=self.p)

    def forward(self, x: torch.Tensor):
        return self.add_noise(x)


def get_augmentor(name, **kwargs):
    if name == "rir":
        return AddRIR(**kwargs)
    if name == "noise":
        return AddNoise(**kwargs)
    if name == "specaugment":
        return SpecAugment(**kwargs)


def get_composed_augmentations(config, sampling_rate: int = 16000):
    augmentations = []
    for key in config.keys():
        augmentations.append(get_augmentor(key, sampling_rate=sampling_rate, **config[key]))
    return Compose(augmentations)
