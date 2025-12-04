import os
import torch
import torchaudio
from torch.utils.data import Dataset
from preprocessing import mel_spectrogram
import glob
import re
from scipy.stats import kurtosis, skew

class DCASE_Dataset(Dataset):
    def __init__(self, root_dir, mode='train', transform=mel_spectrogram, target_seconds=12):
        self.root_dir = root_dir
        self.transform = transform
        self.target_seconds = target_seconds
        
        if mode == 'train':
            self.file_list = glob.glob(os.path.join(root_dir, 'train', '**', '*.wav'), recursive=True)
        elif mode == 'test':
            self.file_list = glob.glob(os.path.join(root_dir, 'test', '**', '*.wav'), recursive=True)

        self.combination_label_map = self._create_combination_labels()
        self.num_classes = len(self.combination_label_map)

    def _get_label_from_path(self, audio_path):
        # ... (기존 코드와 동일) ...
        path_parts = audio_path.replace('\\', '/').split('/')
        machine_type_str = path_parts[-2]
        base_file_name = path_parts[-1]
        
        label_parts = [machine_type_str]

        if machine_type_str == 'fan':
            attr = base_file_name.split('_')[-1].replace('.wav', '')
            label_parts.append(attr)
        elif machine_type_str == 'gearbox':
            attr = base_file_name.split('_')[-1].replace('.wav', '')
            label_parts.append(attr)
        elif machine_type_str == 'valve':
            match1 = re.search(r'_v1pat_(\d+)_', base_file_name)
            if match1:
                label_parts.append(f"v1_{match1.group(1)}")
            match2 = re.search(r'_v2pat_(\d+)', base_file_name)
            if match2:
                label_parts.append(f"v2_{match2.group(1)}")
        elif machine_type_str == 'ToyCar':
            match1 = re.search(r'_car_([A-Z0-9]+)_', base_file_name)
            if match1:
                label_parts.append(match1.group(1))
            match2 = re.search(r'_spd_([A-Z0-9]+)_', base_file_name)
            if match2:
                label_parts.append(match2.group(1))
            match3 = re.search(r'_mic_(\d+)', base_file_name)
            if match3:
                label_parts.append(f"mic{match3.group(1)}")

        return "_".join(label_parts)


    def _create_combination_labels(self):
        all_files = glob.glob(os.path.join(self.root_dir, '**', '*.wav'), recursive=True)
        unique_labels = sorted(list(set([self._get_label_from_path(p) for p in all_files])))
        return {label: i for i, label in enumerate(unique_labels)}

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        audio_path = self.file_list[idx]
        path_parts = audio_path.replace('\\', '/').split('/')
        machine_type_str = path_parts[-2]
        combination_label_str = self._get_label_from_path(audio_path)
        label = self.combination_label_map[combination_label_str]
        is_normal = "normal" in os.path.basename(audio_path)

        waveform, sr = torchaudio.load(audio_path)
        target_length = sr * self.target_seconds
        length = waveform.size(1)
        if length < target_length:
            waveform = torch.cat([waveform, torch.zeros(waveform.size(0), target_length - length)], dim=1)
        else:
            waveform = waveform[:, :target_length]

        wf_mono = waveform[0].numpy()
        rms = torch.sqrt(torch.mean(waveform[0]**2))
        p2p = torch.max(waveform[0]) - torch.min(waveform[0])
        kurt = kurtosis(wf_mono)
        skewness = skew(wf_mono)
        stat_features = torch.tensor([rms, p2p, kurt, skewness], dtype=torch.float)
        
        spec = self.transform(waveform, sr) if self.transform else waveform
        if spec.dim() == 2:
            spec = spec.unsqueeze(0)

        return spec.float(), stat_features, torch.tensor(label, dtype=torch.long), is_normal, machine_type_str