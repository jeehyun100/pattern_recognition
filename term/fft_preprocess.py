# %% md

# Data preprocessing
## extract features and save

# %%

import glob
import librosa
import numpy as np
import sys
import scipy


# %%

# Compute short time Fourier transformation (STFT).
def stft(sig, nfft, win_length_time, hop_length_time, fs, window_type='hann'):
    win_sample = int(win_length_time * fs)
    hop_sample = int(hop_length_time * fs)

    if window_type == 'hann':
        window = np.hanning(win_sample)
    elif window_type == 'hamming':
        window = np.hamming(win_sample)
    else:
        print('Wrong window type : {}'.format(window_type))
        raise StopIteration

    n_frames = int(np.floor((len(sig) - win_sample) / float(hop_sample)) + 1)
    frames = np.stack([window * sig[step * hop_sample: step * hop_sample + win_sample] for step in range(n_frames)])

    stft = np.fft.rfft(frames, n=nfft, axis=1)
    return stft


# Obtain mel-scale filterbank.
def get_melfb(sr, nfft, n_mels):
    mel_fb = librosa.filters.mel(sr, n_fft=nfft, n_mels=n_mels)
    return mel_fb


# Compute log mel spectrogram.
def compute_log_melspectrogram(spec, sr, nfft, n_mels):
    eps = sys.float_info.epsilon
    mel_fb = get_melfb(sr, nfft, n_mels)
    power_spec = spec ** 2
    mel_spec = np.matmul(power_spec, mel_fb.transpose())
    mel_spec = 10 * np.log10(mel_spec + eps)
    return mel_spec


# Compute MFCC.
def compute_mfcc(spec, sr, nfft, n_mels, n_mfcc):
    mel_spec = compute_log_melspectrogram(spec, sr, nfft, n_mels)
    mfcc = scipy.fftpack.dct(mel_spec, axis=-1, norm='ortho')
    return mfcc[:, :n_mfcc]


# %%

def make_dataset(data_list, spk_list, sr, nfft, window_len, hop_len,
                 path='./', win_type='hann', feature_type='fft', n_coeff=64, n_mfcc=13):
    list_x = []
    list_y = []
    for data in data_list:
        sig, _ = librosa.load(data, sr=sr)
        feature = stft(sig, nfft, window_len, hop_len, sr, window_type=win_type)
        feature = abs(feature)
        if feature_type == 'mel':
            feature = compute_log_melspectrogram(feature, sr, nfft, n_coeff)
        elif feature_type == 'mfcc':
            feature = compute_mfcc(feature, sr, nfft, n_coeff, n_mfcc)
        label = spk_list.index(data.split('/')[-1].split('\\')[-1].split('_')[0])
        list_x.append(feature)
        list_y.append(label)
    list_x = np.array(list_x)
    list_y = np.array(list_y)
    np.savez(path, x=list_x, y=list_y)
    print(len(list_x), len(list_y))


# %%

train_list = glob.glob('/Users/yewoo/dev/Speech Interface/proj3/SpeakerDB/Train/*_train.wav')
test_list = glob.glob('/Users/yewoo/dev/Speech Interface/proj3/SpeakerDB/Test/*_test.wav')

# %%

speaker_list = [train_list[i].split('/')[-1].split('\\')[-1].split('_')[0] for i in range(len(train_list))]
with open('/Users/yewoo/dev/Speech Interface/proj3/GMM_Experiments/GMM_speakers.txt', 'w') as f:
    for spk in speaker_list:
        f.write(spk + '\n')

# %%

dataset = make_dataset(train_list, speaker_list, 16000, 1024, 0.025, 0.01,
                       path='./gmm_tr_data_j.npz', feature_type='mfcc')


dataset = make_dataset(test_list, speaker_list, 16000, 1024, 0.025, 0.01,
                       path='./gmm_test_data_j.npz', feature_type='mfcc')