import os
from torch.utils.data import Dataset
import torch
from torch.nn.functional import interpolate
import numpy as np
from batchgenerators.augmentations.normalizations import min_max_normalization


class SpacingDataset(Dataset):
    """
    Dataset for upsampling different simulation spacings.
    """
    def __init__(self, root_dir, low_res, high_res, no_of_spacings=4,
                 data_type="initial_pressure", normalization="min_max"):
        self.root_dir = root_dir
        self.high_res = high_res
        self.low_res = low_res
        self.upscale_factor = int(low_res/high_res)
        self.no_of_spacings = no_of_spacings
        self.volumes = sorted(os.listdir(self.root_dir))
        self.datatype = data_type
        self.normalization = normalization

    def __len__(self):
        return int(len(self.volumes))

    def __getitem__(self, idx):

        volume_path = os.path.join(self.root_dir, self.volumes[idx])
        spacings = os.listdir(volume_path)
        for spacing in spacings:
            if str(self.low_res) in spacing:
                low_res_path = os.path.join(volume_path, spacing)
            elif str(self.high_res) in spacing:
                high_res_path = os.path.join(volume_path, spacing)

        low_res_data = np.load(os.path.join(low_res_path, os.listdir(low_res_path)[0]))[self.datatype]
        high_res_data = np.load(os.path.join(high_res_path, os.listdir(high_res_path)[0]))[self.datatype]

        if self.normalization == "min_max":
            high_res_data, high_res_max, high_res_min = normalize_min_max(high_res_data)
            low_res_data, low_res_max, low_res_min = normalize_min_max(low_res_data)

        elif self.normalization == "standardize":
            high_res_data, high_res_mean, high_res_std = standardize(high_res_data)
            low_res_data, low_res_mean, low_res_std = standardize(low_res_data)

        else:
            high_res_data, high_res_max, high_res_min = high_res_data, np.max(high_res_data), np.min(high_res_data)
            low_res_data, low_res_max, low_res_min = low_res_data, np.max(low_res_data), np.min(low_res_data)

        tensor_hr = torch.from_numpy(high_res_data).unsqueeze(0).type(torch.float32)
        tensor_lr = torch.from_numpy(low_res_data).unsqueeze(0).type(torch.float32)

        if self.normalization == "standardize":
            return {"high_res": tensor_hr, "low_res": tensor_lr,
                    "low_res_mean": low_res_mean, "low_res_std": low_res_std}

        else:
            return {"high_res": tensor_hr, "low_res": tensor_lr,
                    "low_res_max": low_res_max, "low_res_min": low_res_min}


class UpsampleDataset(Dataset):
    """
    Dataset for multispectral initial pressure distributions of photoacoustic images created by mcx-simulation.
    """

    def __init__(self, root_dir, no_of_wavelengths=1, data_type=None,
                 upscale_factor=2, shuffle=False, normalization="min_max"):
        self.root_dir = root_dir
        self.upscale_factor = upscale_factor
        self.no_of_wavelengths = no_of_wavelengths
        self.volumes = sorted(os.listdir(self.root_dir))
        self.datatype = data_type
        self.shuffle = shuffle
        self.sets = ["initial_pressure", "fluence", "mua"]
        self.set_size = self.no_of_wavelengths*len(self.volumes)
        self.normalization = normalization

        if self.shuffle is True:
            self.no_of_sets = 3
        else:
            self.no_of_sets = 1

    def __len__(self):
        return int(self.no_of_sets*self.set_size)

    def __getitem__(self, idx):

        if self.shuffle is True:
            print(idx)
            if idx >= 2*self.set_size:
                self.datatype = "mua"
                idx = idx % (self.sets.index(self.datatype) * self.set_size)
            elif idx >= self.set_size:
                self.datatype = "fluence"
                idx = idx % (self.sets.index(self.datatype) * self.set_size)
            else:
                self.datatype = "initial_pressure"
        if self.datatype == "mua":
            log = False
        else:
            log = True

        volume_no = int(idx/self.no_of_wavelengths)

        volume_path = os.path.join(self.root_dir, self.volumes[volume_no], self.datatype)
        output_file = os.listdir(volume_path)[0]
        data = np.load(os.path.join(volume_path, output_file))

        if self.normalization == "min_max":
            data, low_res_max, low_res_min = normalize_min_max(data, log=log)

        elif self.normalization == "standardize":
            data, low_res_mean, low_res_std = standardize(data, log=log)

        tensor_hr = torch.from_numpy(data).unsqueeze(0).unsqueeze(0)
        tensor_lr = interpolate(tensor_hr,
                                size=(int(data.shape[0]/self.upscale_factor), int(data.shape[1]/self.upscale_factor)))

        tensor_hr = tensor_hr.squeeze(0).type(torch.float32)
        tensor_lr = tensor_lr.squeeze(0).type(torch.float32)

        if self.normalization == "standardize":
            return {"high_res": tensor_hr, "low_res": tensor_lr,
                    "low_res_mean": low_res_mean, "low_res_std": low_res_std}

        else:
            return {"high_res": tensor_hr, "low_res": tensor_lr,
                    "low_res_max": low_res_max, "low_res_min": low_res_min}


class InitialPressureDatasetSRCNN(Dataset):
    """
    Dataset for multispectral initial pressure distributions of photoacoustic images created by mcx-simulation.
    """

    def __init__(self, root_dir, no_of_wavelengths=1, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.no_of_wavelengths = no_of_wavelengths
        self.volumes = sorted(os.listdir(self.root_dir))

    def __len__(self):
        return int(self.no_of_wavelengths*len(self.volumes))

    def __getitem__(self, idx):
        volume_no = int(idx/self.no_of_wavelengths)

        volume_path = os.path.join(self.root_dir, self.volumes[volume_no], "initial_pressure")
        output_file = os.listdir(volume_path)[0]
        initial_pressure = np.load(os.path.join(volume_path, output_file))
        initial_pressure, init_max, init_min = normalize_min_max(initial_pressure)

        tensor_hr = torch.from_numpy(initial_pressure).unsqueeze(0).unsqueeze(0)
        tensor_lr = interpolate(tensor_hr, size=(int(initial_pressure.shape[0]/2), int(initial_pressure.shape[1]/2)))
        tensor_lr = interpolate(tensor_hr,
                                size=(int(initial_pressure.shape[0] + 8 + 4 + 4),
                                      int(initial_pressure.shape[1] + 8 + 4 + 4)),
                                mode="bicubic",
                                align_corners=False)

        tensor_hr = tensor_hr.squeeze(0).type(torch.float32)
        tensor_lr = tensor_lr.squeeze(0).type(torch.float32)

        return {"high_res": tensor_hr, "low_res": tensor_lr, "init_max": init_max, "init_min": init_min, "idx": idx}


def normalize_min_max(data, log=True):
    if log is True:
        data = np.log10(data)
    mx = np.amax(data)
    mn = np.amin(data)
    normalized_data = min_max_normalization(data, eps=1e-16)

    return normalized_data, mx, mn


def normalize_min_max_inverse(data, mx=1, mn=0, log=True):
    if log is True:
        data = 1 - data
        data *= -(mx - mn)
        data += mx
        data = 10 ** data
    else:
        data *= mx - mn
        data += mx

    return data


def standardize(data, log=True):
    if log is True:
        data = np.log10(data)
    mean = np.mean(data)
    std = np.std(data)
    data = (data - mean) / std
    return data, mean, std


def standardize_inverse(data, mean=0, std=1, log=True):
    data = data * std + mean
    if log is True:
        data = 10 ** data
    return data

