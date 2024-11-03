import torch
import pickle
import numpy as np
import pandas as pd
import librosa
import keras
from tqdm import tqdm

# dataset = pd.read_csv("audios_dataset.csv", sep=";")

# train_data = dataset.sample(frac=0.75, random_state=0)
# valid_data = dataset.drop(train_data.index)

# x_train = train_data.drop('genre_id', axis=1)
# y_train = train_data['genre_id']
# x_valid = valid_data.drop('genre_id', axis=1)
# y_valid = valid_data['genre_id']

# y_train = torch.FloatTensor(keras.utils.to_categorical(y_train, 19))
# y_valid = torch.FloatTensor(keras.utils.to_categorical(y_valid, 19))
# y_valid_for_accuracy = torch.FloatTensor(valid_data['genre_id'].to_list())

# pickle.dump(y_train, open('y_train.sav', 'wb'))
# pickle.dump(y_valid, open('y_valid.sav', 'wb'))
# pickle.dump(y_valid_for_accuracy, open('y_valid_for_accuracy.sav', 'wb'))

# # Функция для преобразования спектрограммы в изображение
# def spec_to_image(spec, eps=1e-6):
#   mean = spec.mean()
#   std = spec.std()
#   spec_norm = (spec - mean) / (std + eps)
#   spec_min, spec_max = spec_norm.min(), spec_norm.max()
#   spec_scaled = 255 * (spec_norm - spec_min) / (spec_max - spec_min)
#   spec_scaled = spec_scaled.astype(np.uint8)
#   return spec_scaled

# # Функция для получения мел-спектрограммы из аудиофайла
# def get_melspectrogram_db(file_path, sr=44100, n_fft=2048, hop_length=512, n_mels=128, fmin=20, fmax=8300, top_db=80):
#   wav,sr = librosa.load(file_path, sr=sr)
#   if wav.shape[0]<5*sr:
#     wav=np.pad(wav,int(np.ceil((5*sr-wav.shape[0])/2)),mode='reflect')
#   else:
#     wav=wav[:5*sr]
#   spec=librosa.feature.melspectrogram(y=wav, sr=sr, n_fft=n_fft,
#               hop_length=hop_length,n_mels=n_mels,fmin=fmin,fmax=fmax)
#   spec_db=librosa.power_to_db(spec,top_db=top_db)
#   return spec_db

# X_train = []
# X_valid = []

# for i in tqdm(x_train['audio'].index):
#   spec_tensor = spec_to_image(get_melspectrogram_db(f"audios/audio_{i}.wav"))[np.newaxis,...]
#   X_train.append(spec_tensor)

# for i in tqdm(x_valid['audio'].index):
#   spec_tensor = spec_to_image(get_melspectrogram_db(f"audios/audio_{i}.wav"))[np.newaxis,...]
#   X_valid.append(spec_tensor)

# pickle.dump(X_train, open('X_train.sav', 'wb'))
# pickle.dump(X_valid, open('X_valid.sav', 'wb'))

X_train = pickle.load(open('X_train.sav', 'rb'))[:1001]
X_valid = pickle.load(open('X_valid.sav', 'rb'))[:1001]
X_train = np.array(X_train)
X_valid = np.array(X_valid)
X_train = torch.FloatTensor(X_train)
X_valid = torch.FloatTensor(X_valid)

y_train = pickle.load(open('y_train.sav', 'rb'))[:1001]
y_valid = pickle.load(open('y_valid.sav', 'rb'))[:1001]
y_valid_for_accuracy = pickle.load(open('y_valid_for_accuracy.sav', 'rb'))[:1001]
