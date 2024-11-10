import torch
import torchaudio
import pandas as pd
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder


class AudiosDataset(Dataset):
    def __init__(
        self,
        annotation_file,
        audio_dir,
        num_samples,
        target_sample_rate,
        transform_func,
        device,
        label_encoder,
    ):
        self.annotation_file = pd.read_csv(annotation_file, sep=",")
        self.audio_dir = audio_dir
        self.num_samples = num_samples
        self.target_sample_rate = target_sample_rate
        self.device = device
        self.transform_func = transform_func.to(self.device)
        self.label_encoder = label_encoder

    def __len__(self):
        return len(self.annotation_file)

    def __getitem__(self, index):
        label = self._get_audio_sample_label(index)
        signal, sr = torchaudio.load(self._get_audio_sample_path(index))
        signal = signal.to(self.device)
        signal = self._resample_if_necessary(signal, sr)
        signal = self._mix_down_if_necessary(signal)
        signal = self._cut_if_necessary(signal)
        signal = self._right_pad_if_necessary(signal)
        signal = self.transform_func(signal)

        return signal, label

    def _get_audio_sample_path(self, index):
        return f"{self.audio_dir}/{self.annotation_file.iloc[index, 0]}"

    def _get_audio_sample_label(self, index):
        return int(self.label_encoder.transform([self.annotation_file.iloc[index, 1]])[0])

    def _resample_if_necessary(self, signal, sr):
        if sr != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.target_sample_rate)
            signal = resampler(signal)
        return signal

    def _mix_down_if_necessary(self, signal):
        if signal.shape[0] > 1:
            signal = torch.mean(signal, dim=0, keepdim=True)
        return signal

    def _cut_if_necessary(self, signal):
        if signal.shape[1] > self.num_samples:
            signal = signal[:, : self.num_samples]
        return signal

    def _right_pad_if_necessary(self, signal):
        length_signal = signal.shape[1]
        if length_signal < self.num_samples:
            num_missing_samples = self.num_samples - length_signal
            last_dim_padding = (0, num_missing_samples)  # (left, right)
            signal = torch.nn.functional.pad(signal, last_dim_padding)
        return signal


NUM_SAMPLES = 661500
SAMPLE_RATE = 22050

get_melspectrogram = torchaudio.transforms.MelSpectrogram(
    sample_rate=SAMPLE_RATE, n_fft=1024, hop_length=512, n_mels=64
)

label_encoder = LabelEncoder()
label_encoder.fit(
    [
        "blues",
        "classical",
        "country",
        "disco",
        "hiphop",
        "jazz",
        "metal",
        "pop",
        "reggae",
        "rock",
    ]
)

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
print(f"Using device: ", device)

train_dataset = AudiosDataset(
    "GT-Music-Genre/train.csv",
    "GT-Music-Genre",
    NUM_SAMPLES,
    SAMPLE_RATE,
    get_melspectrogram,
    device,
    label_encoder,
)

valid_dataset = AudiosDataset(
    "GT-Music-Genre/valid.csv",
    "GT-Music-Genre",
    NUM_SAMPLES,
    SAMPLE_RATE,
    get_melspectrogram,
    device,
    label_encoder,
)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True)
valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=128, shuffle=False)
