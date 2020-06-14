import numpy as np
import tensorflow as tf
import glob
import pandas as pd
import matplotlib.pyplot as plt
import os
import sklearn.mixture as mixture
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
#from sklearn.svm import SVC
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
#from sklearn.metrics import accuracy_score
import scipy.fftpack as fftpack
import librosa
import sys
from sklearn import cluster


vq_cluster = 5
init_param_0 = np.array([0.02095179, 0.90566462, 0.43209739, -0.25544495, -0.77559784])
init_param_1 = np.array([0.02232163, 0.43481753, -0.76255045, 0.9057137, -0.25071883])
init_param_2 = np.array([-0.25389706, 0.43028382, 0.02092518, 0.90301782, -0.76785306])
init_param_3 = np.array([0.02219885, 0.90365064, -0.24924463, 0.43321321, -0.75605498])
init_param_4 = np.array([-0.25172194, 0.89856953, 0.41381263, 0.01912027, -0.75912962])
init_param_5 = np.array([0.01752499, 0.896993, -0.76920107, -0.25543445, 0.4079556])


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
    #
    # stft_frames = [ fftpack.fft(x,N) for x in frames]
    freq_axis = np.linspace(0,fs,n_frames)
    # return(stft_frames, freq_axis)

    return stft, freq_axis

def get_log_mel_transform(p_data):
    #_tmp = train_x[0, :, 0]
    p_s = p_data[:,0]
    nfft = 512
    window_len = 0.4#0.3#0.5  # 1.0#0.3
    hop_len = 0.1#0.1  # 0.5
    sr = 50
    win_type = 'hann'
    n_coeff = 8

    p_s = p_data[:,0]
    p_stft, freq_axis = stft(p_s, nfft, window_len, hop_len, sr, window_type=win_type)
    p_stft = abs(p_stft)
    feature1 = compute_log_melspectrogram(p_stft, sr, nfft, n_coeff)
    p_s = p_data[:,1]
    p_stft, freq_axis = stft(p_s, nfft, window_len, hop_len, sr, window_type=win_type)
    p_stft = abs(p_stft)
    feature2 = compute_log_melspectrogram(p_stft, sr, nfft, n_coeff)
    p_s = p_data[:,2]
    p_stft, freq_axis = stft(p_s, nfft, window_len, hop_len, sr, window_type=win_type)
    p_stft = abs(p_stft)
    feature3 = compute_log_melspectrogram(p_stft, sr, nfft, n_coeff)

    p_s = p_data[:, 3]
    p_stft, freq_axis = stft(p_s, nfft, window_len, hop_len, sr, window_type=win_type)
    p_stft = abs(p_stft)
    feature4 = compute_log_melspectrogram(p_stft, sr, nfft, n_coeff)
    p_s = p_data[:, 4]
    p_stft, freq_axis = stft(p_s, nfft, window_len, hop_len, sr, window_type=win_type)
    p_stft = abs(p_stft)
    feature5 = compute_log_melspectrogram(p_stft, sr, nfft, n_coeff)
    p_s = p_data[:, 5]
    p_stft, freq_axis = stft(p_s, nfft, window_len, hop_len, sr, window_type=win_type)
    p_stft = abs(p_stft)
    feature6 = compute_log_melspectrogram(p_stft, sr, nfft, n_coeff)

    feature = np.concatenate((feature1,feature2,feature3,feature4,feature5,feature6),axis=1)


    return feature.flatten()


def load_datasets_by_label(tr_data, te_data, grid_search=False, label = None, type='flatten'):#, activity_class="ALL", one_hot=False):
    """selete data by label

    Args:
        data_path (string) : Path data load
        dataset_name : For seleced dataset name  ['p1', 'p2']

    Returns:
        2-D Array: P1 Train data
        2-D Array: P1 Test data
        2-D Array: P2 Train data
        2-D Array: P2 Test data

    """
    tr_data_by_label = tr_data
    te_data_by_label = te_data
    #tr_data_by_label = tr_data_by_label[np.where(tr_data_by_label[:, 0] == 1)][:, 3]
    #flatten_arr = [x.flatten()for x in tr_data_by_label]
    if label is not None:
        tr_data_by_label = tr_data[np.where(tr_data[:,0]==str(label))]
        te_data_by_label = te_data[np.where(te_data[:,0]==str(label))]
        #np_all_data[np.where(np_all_data[:, 0] == 1)][:, 3]

    if 'flatten' in type:
        train_x = [x.flatten() for x in tr_data_by_label[:,3]]
        train_y = tr_data_by_label[:,0]
        test_x = [x.flatten() for x in te_data_by_label[:,3]]
        test_y = te_data_by_label[:,0]
    elif 'raw' in type:
        #train_x = tr_data_by_label[:,3]
        train_x = np.array([x for x in tr_data_by_label[:,3]])
        train_y = tr_data_by_label[:,0:1].astype(int)-1
        test_x =  np.array([x for x in te_data_by_label[:,3]])
        test_y = te_data_by_label[:,0:1].astype(int)-1
    elif 'fft' in type:
        #train_x = tr_data_by_label[:,3]
        train_x_s = np.array([get_log_mel_transform(x) for x in tr_data_by_label[:,3]])
        train_x = np.squeeze(train_x_s)
        train_y = tr_data_by_label[:,0:1].astype(int)-1
        test_x_s =  np.array([get_log_mel_transform(x) for x in te_data_by_label[:,3]])
        test_x = np.squeeze(test_x_s)
        test_y = te_data_by_label[:,0:1].astype(int)-1
        del train_x_s
        del test_x_s

    return train_x, train_y, test_x, test_y



    # if grid_search:
    #     return tr_data_by_label[:, :6], tr_data_by_label[:, 6], te_data_by_label[:, :6], te_data_by_label[:, 6]
    # else:
    #     return tr_data_by_label[:,:6], tr_data_by_label[:,6:7], te_data_by_label[:,:6], te_data_by_label[:,6:7]

def load_datasets(data_path, npz=None , shuffle= False):#, activity_class="ALL", one_hot=False):
    """Load dataset unsing numpy wih data path
        . Datasets :
          . filename : 0009_01_5.txt
            “aaaa_bb_c.txt,” where aaaa is the file ID, bb is the user ID, and c is the ground truth activity class
          . data : The six columns represent accelerations (in standard gravity unit g=9.8m/s2) in X, Y, and Z
                   directions, and angular velocities (in rad/sec) in X, Y, and Z directions

    Args:
        data_path (string) : Path data load
        dataset_name : For seleced dataset name  ['p1', 'p2']

    Returns:
        2-D Array: P1 Train data
        2-D Array: P1 Test data
        2-D Array: P2 Train data
        2-D Array: P2 Test data

    """

    if npz == None:
        cnt = 0
        np_all_data = np.empty(shape=[0,9])
        rowlist = list()
        for file in glob.glob(data_path + "/*.txt"):
            #if str(activity_class) in os.path.basename(os.path.splitext(file)[0]).split("_")[2]:
                #print(file)
            col_list = list()
            cnt += 1
            row = np.loadtxt(file)
            file_index, user_id, activity_class = os.path.basename(os.path.splitext(file)[0]).split("_")
            #np_lables = np.array([activity_class, user_id,file_index])
            col_list.append(str(activity_class))
            col_list.append(int(user_id))
            col_list.append(int(file_index))
            col_list.append(row)

            # no_lables_reshape = np.repeat([np_lables],row.shape[0], 0)
            # np_p1_tr_a = np.column_stack([row, no_lables_reshape])

            rowlist.append(col_list)
            del col_list

            if cnt % 100 == 0:
                print(cnt)
            #     break
        np_all_data = np.array(rowlist)
        np.savez("./npz/activity_nparray2.npz", data = np_all_data)
        print("save complete")
    else:
        np_all_data = np.load("./npz/activity_nparray2.npz",allow_pickle=True)['data']
    tr_data, te_data,  = train_test_split(np_all_data,test_size=0.3,shuffle=shuffle,random_state=1004)

    return tr_data, te_data



# Compute log mel spectrogram.
def compute_log_melspectrogram(spec, sr, nfft, n_mels):
    eps = sys.float_info.epsilon
    mel_fb = get_melfb(sr, nfft, n_mels)
    power_spec = spec**2
    mel_spec = np.matmul(power_spec, mel_fb.transpose())
    mel_spec = 10*np.log10(mel_spec+eps)
    return mel_spec

# Obtain mel-scale filterbank.
def get_melfb(sr, nfft, n_mels):
    mel_fb = librosa.filters.mel(sr, n_fft=nfft, n_mels=n_mels)
    return mel_fb



def plot_graph(tr_data, te_data):

    labels = ['Walking','Walking upstairs','Walking downstairs','Sitting','Standing','Laying']


    for i, label in enumerate(labels):
        x_train, y_train, x_test, y_test = load_datasets_by_label(tr_data, te_data,  label=i+1, type='raw')

        sr = 50
        _tmp1 = x_train[0,:,0]
        _tmp2 = x_train[0,:,1]
        _tmp3 = x_train[0,:,2]
        _tmp4 = x_train[0,:,3]
        _tmp5 = x_train[0,:,4]
        _tmp6 = x_train[0,:,5]
        # We'll use the numpy function "linspace" to create a time axis for plotting
        p_timeaxis = np.linspace(0, (1 / sr * _tmp1.shape[0]), len(_tmp1))

        # And plot them
        plt.figure(1, figsize=(10, 4))

        #plt.subplot(211)
        plt.plot(p_timeaxis, _tmp1)
        plt.plot(p_timeaxis, _tmp2)
        plt.plot(p_timeaxis, _tmp3)
        plt.plot(p_timeaxis, _tmp4)
        plt.plot(p_timeaxis, _tmp5)
        plt.plot(p_timeaxis, _tmp6)
        yminmax = (x_train[0, :, :].min(), x_train[0,:,:].max())
        #ymin, ymax = min(_tmp6),
        plt.ylim(yminmax[0], yminmax[1])
        #plt.xlim(0, 10)

        plt.title(label)
        plt.show()

        #p_s = p_data[:, 0]
        nfft = 512
        window_len = 0.4  # 0.3#0.5  # 1.0#0.3
        hop_len = 0.1  # 0.1  # 0.5
        sr = 50
        win_type = 'hann'
        n_coeff = 8

        p_s = _tmp1
        p_stft, freq_axis = stft(p_s, nfft, window_len, hop_len, sr, window_type=win_type)
        p_stft = abs(p_stft)
        feature1 = compute_log_melspectrogram(p_stft, sr, nfft, n_coeff)
        p_s = _tmp2
        p_stft, freq_axis = stft(p_s, nfft, window_len, hop_len, sr, window_type=win_type)
        p_stft = abs(p_stft)
        feature2 = compute_log_melspectrogram(p_stft, sr, nfft, n_coeff)
        p_s = _tmp3
        p_stft, freq_axis = stft(p_s, nfft, window_len, hop_len, sr, window_type=win_type)
        p_stft = abs(p_stft)
        feature3 = compute_log_melspectrogram(p_stft, sr, nfft, n_coeff)

        p_s = _tmp4
        p_stft, freq_axis = stft(p_s, nfft, window_len, hop_len, sr, window_type=win_type)
        p_stft = abs(p_stft)
        feature4 = compute_log_melspectrogram(p_stft, sr, nfft, n_coeff)
        p_s = _tmp5
        p_stft, freq_axis = stft(p_s, nfft, window_len, hop_len, sr, window_type=win_type)
        p_stft = abs(p_stft)
        feature5 = compute_log_melspectrogram(p_stft, sr, nfft, n_coeff)
        p_s = _tmp6
        p_stft, freq_axis = stft(p_s, nfft, window_len, hop_len, sr, window_type=win_type)
        p_stft = abs(p_stft)
        feature6 = compute_log_melspectrogram(p_stft, sr, nfft, n_coeff)

        plt.figure(figsize=(10, 4))
        plt.plot(feature1)
        plt.plot(feature2)
        plt.plot(feature3)
        plt.plot(feature4)
        plt.plot(feature5)
        plt.plot(feature6)

        #        plt.colorbar(format='%+2.0f dB')
        plt.title('Mel-Spectrogram ' + label)

        # plt.savefig('Mel-Spectrogram example.png')
        plt.show()

        print("donw")

        p_s = _tmp1
        k_means = cluster.KMeans(n_clusters=vq_cluster, init=init_param_0.reshape((-1, 1)), n_init=1, verbose=2)
        X = p_s.reshape((-1, 1))
        k_means.fit(X)
        values = k_means.cluster_centers_.squeeze()
        labels = k_means.labels_
        vq_feature1 = np.choose(labels, values)
        vq_feature1.shape = p_s.shape

        p_s = _tmp2
        k_means = cluster.KMeans(n_clusters=vq_cluster, init=init_param_1.reshape((-1, 1)), n_init=1, verbose=2)
        X = p_s.reshape((-1, 1))
        k_means.fit(X)
        values = k_means.cluster_centers_.squeeze()
        labels = k_means.labels_
        vq_feature2 = np.choose(labels, values)
        vq_feature2.shape = p_s.shape

        p_s = _tmp3
        k_means = cluster.KMeans(n_clusters=vq_cluster, init=init_param_2.reshape((-1, 1)), n_init=1, verbose=2)
        X = p_s.reshape((-1, 1))
        k_means.fit(X)
        values = k_means.cluster_centers_.squeeze()
        labels = k_means.labels_
        vq_feature3 = np.choose(labels, values)
        vq_feature3.shape = p_s.shape

        p_s = _tmp4
        k_means = cluster.KMeans(n_clusters=vq_cluster, init=init_param_3.reshape((-1, 1)), n_init=1, verbose=2)
        X = p_s.reshape((-1, 1))
        k_means.fit(X)
        values = k_means.cluster_centers_.squeeze()
        labels = k_means.labels_
        vq_feature4 = np.choose(labels, values)
        vq_feature4.shape = p_s.shape

        p_s = _tmp5
        k_means = cluster.KMeans(n_clusters=vq_cluster, init=init_param_4.reshape((-1, 1)), n_init=1, verbose=2)
        X = p_s.reshape((-1, 1))
        k_means.fit(X)
        values = k_means.cluster_centers_.squeeze()
        labels = k_means.labels_
        vq_feature5 = np.choose(labels, values)
        vq_feature5.shape = p_s.shape

        p_s = _tmp6
        k_means = cluster.KMeans(n_clusters=vq_cluster, init=init_param_5.reshape((-1, 1)), n_init=1, verbose=2)
        X = p_s.reshape((-1, 1))
        k_means.fit(X)
        values = k_means.cluster_centers_.squeeze()
        labels = k_means.labels_
        vq_feature6 = np.choose(labels, values)
        vq_feature6.shape = p_s.shape

        _tmp_all = vq_feature1 + vq_feature2 + vq_feature3 + vq_feature4 + vq_feature5 + vq_feature6


        plt.figure(figsize=(10, 4))
        #plt.ylim((_tmp_all.min(), _tmp_all.max()))
        plt.plot(vq_feature1)
        plt.plot(_tmp1)
        plt.plot(vq_feature2)
        plt.plot(_tmp2)
        plt.plot(vq_feature3)
        plt.plot(_tmp3)
        plt.plot(vq_feature4)
        plt.plot(_tmp4)
        plt.plot(vq_feature5)
        plt.plot(_tmp5)
        plt.plot(vq_feature6)
        plt.plot(_tmp6)



        #        plt.colorbar(format='%+2.0f dB')
        plt.title('VQ ' + label)

        # plt.savefig('Mel-Spectrogram example.png')
        plt.show()


if __name__ == "__main__":
    os.makedirs("./csv/", exist_ok=True)
    os.makedirs("./plot/", exist_ok=True)
    os.makedirs("./npz/", exist_ok=True)
    dataset = ['1', '2', '3', '4', '5', '6']
    #dataset = ['1', '2']

    data_path = "./mlpr20_project_train_data"

    tr_data, te_data = load_datasets(data_path, False, shuffle= True)
    #plot_graph()qq
    plot_graph(tr_data, te_data )
    #exam_all_train(tr_data, te_data )
    #for i in range(6):
    #load_datasets(data_path, True)