#%%

# All Includes

import numpy as np
import matplotlib
import matplotlib.pyplot as pltåç
import tensorflow as tf  # Version 1.0.0 (some previous versions are used in past commits)
from sklearn import metrics

import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import scipy.fftpack as fftpack
import librosa
import sys
from sklearn import metrics
from sklearn import cluster


# Useful Constants

# Those are separate normalised input features for the neural network
INPUT_SIGNAL_TYPES = [
    "body_acc_x_",
    "body_acc_y_",
    "body_acc_z_",
    "body_gyro_x_",
    "body_gyro_y_",
    "body_gyro_z_",
]


# Output classes to learn how to classify
LABELS = ['Walking', 'Walking upstairs', 'Walking downstairs', 'Sitting', 'Standing', 'Laying']
#
# TRAIN = "train/"
# TEST = "test/"
# DATA_PATH = "./data/"
# DATASET_PATH = DATA_PATH + "UCI HAR Dataset/"
#
#
# # Load "X" (the neural network's training and testing inputs)
#
# def load_X(X_signals_paths):
#     X_signals = []
#
#     for signal_type_path in X_signals_paths:
#         file = open(signal_type_path, 'r')
#         # Read dataset from disk, dealing with text files' syntax
#         X_signals.append(
#             [np.array(serie, dtype=np.float32) for serie in [
#                 row.replace('  ', ' ').strip().split(' ') for row in file
#             ]]
#         )
#         file.close()
#
#     return np.transpose(np.array(X_signals), (1, 2, 0))
#
#
# X_train_signals_paths = [
#     DATASET_PATH + TRAIN + "Inertial Signals/" + signal + "train.txt" for signal in INPUT_SIGNAL_TYPES
# ]
# X_test_signals_paths = [
#     DATASET_PATH + TEST + "Inertial Signals/" + signal + "test.txt" for signal in INPUT_SIGNAL_TYPES
# ]
#
# # X_train = load_X(X_train_signals_paths)
# # X_test = load_X(X_test_signals_paths)
#
#
# # Load "y" (the neural network's training and testing outputs)
#
# def load_y(y_path):
#     file = open(y_path, 'r')
#     # Read dataset from disk, dealing with text file's syntax
#     y_ = np.array(
#         [elem for elem in [
#             row.replace('  ', ' ').strip().split(' ') for row in file
#         ]],
#         dtype=np.int32
#     )
#     file.close()
#
#     # Substract 1 to each output class for friendly 0-based indexing
#     return y_ - 1

n_classes = 6
vq_cluster = 5
init_param_0 = np.array([0.02095179, 0.90566462, 0.43209739, -0.25544495, -0.77559784])
init_param_1 = np.array([0.02232163, 0.43481753, -0.76255045, 0.9057137, -0.25071883])
init_param_2 = np.array([-0.25389706, 0.43028382, 0.02092518, 0.90301782, -0.76785306])
init_param_3 = np.array([0.02219885, 0.90365064, -0.24924463, 0.43321321, -0.75605498])
init_param_4 = np.array([-0.25172194, 0.89856953, 0.41381263, 0.01912027, -0.75912962])
init_param_5 = np.array([0.01752499, 0.896993, -0.76920107, -0.25543445, 0.4079556])
n_hidden = 16  # Hidden layer num of features
#n_classes = 6  # Total classes (should go up, or should go down)

learning_rate = 0.0025
lambda_loss_amount = 0.0015

batch_size = 1500
display_iter = 30000  # To show test set accuracy during training

#log_cnt = 0

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

def get_log_mel_transform(p_data, type = None):
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


    print("fft converting")
    feature = np.concatenate((feature1, feature2, feature3, feature4, feature5, feature6), axis=1)
    if type != "RNN":
        feature = feature.flatten()


    return feature


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
        train_y = tr_data_by_label[:,0:1].astype(int)-1
        test_x = [x.flatten() for x in te_data_by_label[:,3]]
        test_y = te_data_by_label[:,0:1].astype(int)-1
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
    elif 'rnnftt' in type:
        #train_x = tr_data_by_label[:,3]
        train_x_s = np.array([get_log_mel_transform(x,"RNN") for x in tr_data_by_label[:,3]])
        train_x = np.squeeze(train_x_s)
        train_y = tr_data_by_label[:,0:1].astype(int)-1
        test_x_s =  np.array([get_log_mel_transform(x, "RNN") for x in te_data_by_label[:,3]])
        test_x = np.squeeze(test_x_s)
        test_y = te_data_by_label[:,0:1].astype(int)-1
        del train_x_s
        del test_x_s
    elif 'vq' == type:
        train_x_s = np.array([get_vq(x,) for x in tr_data_by_label[:, 3]])
        train_x = np.squeeze(train_x_s)
        train_y = tr_data_by_label[:,0:1].astype(int)-1
        test_x_s =  np.array([get_vq(x) for x in te_data_by_label[:,3]])
        test_x = np.squeeze(test_x_s)
        test_y = te_data_by_label[:,0:1].astype(int)-1
        del train_x_s
        del test_x_s
    elif 'rnnvq' == type:
        train_x_s = np.array([get_vq(x,"RNN") for x in tr_data_by_label[:, 3]])
        train_x = np.squeeze(train_x_s)
        train_y = tr_data_by_label[:,0:1].astype(int)-1
        test_x_s =  np.array([get_vq(x,"RNN") for x in te_data_by_label[:,3]])
        test_x = np.squeeze(test_x_s)
        test_y = te_data_by_label[:,0:1].astype(int)-1
        del train_x_s
        del test_x_s


    return train_x, train_y, test_x, test_y


def get_vq(p_data, type= None):
    """
    make init centroid point for vq
    :param p_data:
    :return:
    """

    # fit # load
    # _tmp = train_x[0, :, 0]
    #vq_cluster



    p_s = p_data[:,0]
    k_means = cluster.KMeans(n_clusters=vq_cluster, init=init_param_0.reshape((-1,1)), n_init=1, verbose = 2)
    X = p_s.reshape((-1, 1))
    k_means.fit(X)
    values = k_means.cluster_centers_.squeeze()
    labels = k_means.labels_
    vq_feature1 = np.choose(labels, values)
    vq_feature1.shape = p_s.shape

    p_s = p_data[:,1]
    k_means = cluster.KMeans(n_clusters=vq_cluster, init=init_param_1.reshape((-1,1)), n_init=1, verbose = 2)
    X = p_s.reshape((-1, 1))
    k_means.fit(X)
    values = k_means.cluster_centers_.squeeze()
    labels = k_means.labels_
    vq_feature2 = np.choose(labels, values)
    vq_feature2.shape = p_s.shape

    p_s = p_data[:,2]
    k_means = cluster.KMeans(n_clusters=vq_cluster, init=init_param_2.reshape((-1,1)), n_init=1, verbose = 2)
    X = p_s.reshape((-1, 1))
    k_means.fit(X)
    values = k_means.cluster_centers_.squeeze()
    labels = k_means.labels_
    vq_feature3 = np.choose(labels, values)
    vq_feature3.shape = p_s.shape

    p_s = p_data[:,3]
    k_means = cluster.KMeans(n_clusters=vq_cluster, init=init_param_3.reshape((-1,1)), n_init=1, verbose = 2)
    X = p_s.reshape((-1, 1))
    k_means.fit(X)
    values = k_means.cluster_centers_.squeeze()
    labels = k_means.labels_
    vq_feature4 = np.choose(labels, values)
    vq_feature4.shape = p_s.shape

    p_s = p_data[:,4]
    k_means = cluster.KMeans(n_clusters=vq_cluster, init=init_param_4.reshape((-1,1)), n_init=1, verbose = 2)
    X = p_s.reshape((-1, 1))
    k_means.fit(X)
    values = k_means.cluster_centers_.squeeze()
    labels = k_means.labels_
    vq_feature5 = np.choose(labels, values)
    vq_feature5.shape = p_s.shape

    p_s = p_data[:,5]
    k_means = cluster.KMeans(n_clusters=vq_cluster, init=init_param_5.reshape((-1,1)), n_init=1, verbose = 2)
    X = p_s.reshape((-1, 1))
    k_means.fit(X)
    values = k_means.cluster_centers_.squeeze()
    labels = k_means.labels_
    vq_feature6 = np.choose(labels, values)
    vq_feature6.shape = p_s.shape
    #feature = np.concatenate((vq_feature1, vq_feature2, vq_feature3, vq_feature4, vq_feature5, vq_feature6))

    if type == "RNN":
        feature = np.vstack((vq_feature1, vq_feature2, vq_feature3, vq_feature4, vq_feature5, vq_feature6)).T
    else:
        feature = np.concatenate((vq_feature1, vq_feature2, vq_feature3, vq_feature4, vq_feature5, vq_feature6))

    # # And plot them
    # f1 = plt.figure(1, figsize=(14, 4))
    # plt.subplot(211)
    # plt.plot(face_compressed)
    # plt.plot(p_s)
    # plt.title('wav file plot')
    # plt.show()
    #
    # # p_stft, freq_axis = stft(p_s, nfft, window_len, hop_len, sr, window_type=win_type)
    # # p_stft = abs(p_stft)
    # # feature1 = compute_log_melspectrogram(p_stft, sr, nfft, n_coeff)
    # # p_s = p_data[:, 1]
    # # p_stft, freq_axis = stft(p_s, nfft, window_len, hop_len, sr, window_type=win_type)
    # # p_stft = abs(p_stft)
    # # feature2 = compute_log_melspectrogram(p_stft, sr, nfft, n_coeff)
    # # p_s = p_data[:, 2]
    # # p_stft, freq_axis = stft(p_s, nfft, window_len, hop_len, sr, window_type=win_type)
    # # p_stft = abs(p_stft)
    # # feature3 = compute_log_melspectrogram(p_stft, sr, nfft, n_coeff)
    # #
    # # p_s = p_data[:, 3]
    # # p_stft, freq_axis = stft(p_s, nfft, window_len, hop_len, sr, window_type=win_type)
    # # p_stft = abs(p_stft)
    # # feature4 = compute_log_melspectrogram(p_stft, sr, nfft, n_coeff)
    # # p_s = p_data[:, 4]
    # # p_stft, freq_axis = stft(p_s, nfft, window_len, hop_len, sr, window_type=win_type)
    # # p_stft = abs(p_stft)
    # # feature5 = compute_log_melspectrogram(p_stft, sr, nfft, n_coeff)
    # # p_s = p_data[:, 5]
    # # p_stft, freq_axis = stft(p_s, nfft, window_len, hop_len, sr, window_type=win_type)
    # # p_stft = abs(p_stft)
    # # feature6 = compute_log_melspectrogram(p_stft, sr, nfft, n_coeff)
    # #
    # # feature = np.concatenate((feature1, feature2, feature3, feature4, feature5, feature6), axis=1)

    return feature


def get_vg_train(p_data):
    """
    make init centroid point for vq
    :param p_data:
    :return:
    """

    # fit # load
    # _tmp = train_x[0, :, 0]
    #vq_cluster
    all_centroid_init = dict()
    for i in range(6):
        list_all_data = []
        for x in p_data:
            list_all_data.extend(x[:,i])
        all_data_np = np.array(list_all_data)
        k_means = cluster.KMeans(n_clusters=vq_cluster, n_init=4, verbose = 2)
        X = all_data_np.reshape((-1,1))
        k_means.fit(X)

        values = k_means.cluster_centers_.squeeze()
        all_centroid_init[i] = values
        #all_centroid_init.append(values)
    print(all_centroid_init)

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




def LSTM_RNN(_X, _weights, _biases, n_input, n_steps):
    # Function returns a tensorflow LSTM (RNN) artificial neural network from given parameters.
    # Moreover, two LSTM cells are stacked which adds deepness to the neural network.
    # Note, some code of this notebook is inspired from an slightly different
    # RNN architecture used on another dataset, some of the credits goes to
    # "aymericdamien" under the MIT license.

    # (NOTE: This step could be greatly optimised by shaping the dataset once
    # input shape: (batch_size, n_steps, n_input)
    _X = tf.transpose(_X, [1, 0, 2])  # permute n_steps and batch_size
    # Reshape to prepare input to hidden activation
    _X = tf.reshape(_X, [-1, n_input])
    # new shape: (n_steps*batch_size, n_input)

    # ReLU activation, thanks to Yu Zhao for adding this improvement here:
    _X = tf.nn.relu(tf.matmul(_X, _weights['hidden']) + _biases['hidden'])
    # Split data because rnn cell needs a list of inputs for the RNN inner loop
    _X = tf.split(_X, n_steps, 0)
    # new shape: n_steps * (batch_size, n_hidden)

    # Define two stacked LSTM cells (two recurrent layers deep) with tensorflow
    lstm_cell_1 = tf.contrib.rnn.BasicLSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)
    lstm_cell_2 = tf.contrib.rnn.BasicLSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)
    lstm_cells = tf.contrib.rnn.MultiRNNCell([lstm_cell_1, lstm_cell_2], state_is_tuple=True)
    # Get LSTM cell output
    outputs, states = tf.contrib.rnn.static_rnn(lstm_cells, _X, dtype=tf.float32)

    # Get last time step's output feature for a "many-to-one" style classifier,
    # as in the image describing RNNs at the top of this page
    lstm_last_output = outputs[-1]

    # Linear activation
    return tf.matmul(lstm_last_output, _weights['out']) + _biases['out']


def extract_batch_size(_train, step, batch_size):
    # Function to fetch a "batch_size" amount of data from "(X|y)_train" data.

    shape = list(_train.shape)
    shape[0] = batch_size
    batch_s = np.empty(shape)

    for i in range(batch_size):
        # Loop index
        index = ((step - 1) * batch_size + i) % len(_train)
        batch_s[i] = _train[index]

    return batch_s


def one_hot(y_, n_classes=n_classes):
    # Function to encode neural one-hot output labels from number indexes
    # e.g.:
    # one_hot(y_=[[5], [0], [3]], n_classes=6):
    #     return [[0, 0, 0, 0, 0, 1], [1, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0]]

    y_ = y_.reshape(len(y_))
    return np.eye(n_classes)[np.array(y_, dtype=np.int32)]  # Returns FLOATS


def rnn_train():
#    y_train_path = DATASET_PATH + TRAIN + "y_train.txt"
 #   y_test_path = DATASET_PATH + TEST + "y_test.txt"

    # y_train = load_y(y_train_path)
    # y_test = load_y(y_test_path)
    data_path = "./mlpr20_project_train_data"
    tr_data, te_data = load_datasets(data_path, False, shuffle=True)
    X_train, y_train, X_test, y_test = load_datasets_by_label(tr_data, te_data, label=None, type='raw')

    # Input Data

    training_data_count = len(X_train)  # 7352 training series (with 50% overlap between each serie)
    test_data_count = len(X_test)  # 2947 testing series
    n_steps = len(X_train[0])  # 128 timesteps per series
    n_input = len(X_train[0][0])  # 9 input parameters per timestep

    # LSTM Neural Network's internal structure

    # n_hidden = 32  # Hidden layer num of features
    # n_classes = 6  # Total classes (should go up, or should go down)

    # Training
    training_iters = training_data_count * 300  # Loop 300 times on the dataset



    # Some debugging info

    print("Some useful info to get an insight on dataset's shape and normalisation:")
    print("(X shape, y shape, every X's mean, every X's standard deviation)")
    print(X_test.shape, y_test.shape, np.mean(X_test), np.std(X_test))
    print("The dataset is therefore properly normalised, as expected, but not yet one-hot encoded.")


    # Graph input/output
    x = tf.placeholder(tf.float32, [None, n_steps, n_input])
    y = tf.placeholder(tf.float32, [None, n_classes])

    # Graph weights
    weights = {
        'hidden': tf.Variable(tf.random_normal([n_input, n_hidden])), # Hidden layer weights
        'out': tf.Variable(tf.random_normal([n_hidden, n_classes], mean=1.0))
    }
    biases = {
        'hidden': tf.Variable(tf.random_normal([n_hidden])),
        'out': tf.Variable(tf.random_normal([n_classes]))
    }

    pred = LSTM_RNN(x, weights, biases, n_input, n_steps)

    # Loss, optimizer and evaluation
    l2 = lambda_loss_amount * sum(
        tf.nn.l2_loss(tf_var) for tf_var in tf.trainable_variables()
    ) # L2 loss prevents this overkill neural network to overfit the data
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=pred)) + l2 # Softmax loss
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost) # Adam Optimizer

    correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    # To keep track of training's performance
    test_losses = []
    test_accuracies = []
    train_losses = []
    train_accuracies = []

    # Launch the graph
    sess = tf.InteractiveSession(config=tf.ConfigProto(log_device_placement=True))
    init = tf.global_variables_initializer()
    sess.run(init)

    # Perform Training steps with "batch_size" amount of example data at each loop
    step = 1
    while step * batch_size <= training_iters:
        batch_xs = extract_batch_size(X_train, step, batch_size)
        batch_ys = one_hot(extract_batch_size(y_train, step, batch_size))

        # Fit training using batch data
        _, loss, acc = sess.run(
            [optimizer, cost, accuracy],
            feed_dict={
                x: batch_xs,
                y: batch_ys
            }
        )
        train_losses.append(loss)
        train_accuracies.append(acc)

        # Evaluate network only at some steps for faster training:
        if (step * batch_size % display_iter == 0) or (step == 1) or (step * batch_size > training_iters):
            # To not spam console, show training accuracy/loss in this "if"
            print("Training iter #" + str(step * batch_size) + \
                  ":   Batch Loss = " + "{:.6f}".format(loss) + \
                  ", Accuracy = {}".format(acc))

            # Evaluation on the test set (no learning made here - just evaluation for diagnosis)
            loss, acc = sess.run(
                [cost, accuracy],
                feed_dict={
                    x: X_test,
                    y: one_hot(y_test)
                }
            )
            test_losses.append(loss)
            test_accuracies.append(acc)
            print("PERFORMANCE ON TEST SET: " + \
                  "Batch Loss = {}".format(loss) + \
                  ", Accuracy = {}".format(acc))

        step += 1

    print("Optimization Finished!")

    # Accuracy for test data

    one_hot_predictions, accuracy, final_loss = sess.run(
        [pred, accuracy, cost],
        feed_dict={
            x: X_test,
            y: one_hot(y_test)
        }
    )

    test_losses.append(final_loss)
    test_accuracies.append(accuracy)

    draw_plot(one_hot_predictions, final_loss, accuracy, train_losses, train_accuracies, test_losses, test_accuracies, training_iters, y_test)


    # print("FINAL RESULT: " + \
    #       "Batch Loss = {}".format(final_loss) + \
    #       ", Accuracy = {}".format(accuracy))
    #
    #
    #
    # font = {
    #     'family' : 'Bitstream Vera Sans',
    #     'weight' : 'bold',
    #     'size'   : 18
    # }
    # matplotlib.rc('font', **font)
    #
    # width = 12
    # height = 12
    # plt.figure(figsize=(width, height))
    #
    # indep_train_axis = np.array(range(batch_size, (len(train_losses)+1)*batch_size, batch_size))
    # plt.plot(indep_train_axis, np.array(train_losses),     "b--", label="Train losses")
    # plt.plot(indep_train_axis, np.array(train_accuracies), "g--", label="Train accuracies")
    #
    # indep_test_axis = np.append(
    #     np.array(range(batch_size, len(test_losses)*display_iter, display_iter)[:-1]),
    #     [training_iters]
    # )
    # plt.plot(indep_test_axis, np.array(test_losses),     "b-", label="Test losses")
    # plt.plot(indep_test_axis, np.array(test_accuracies), "g-", label="Test accuracies")
    #
    # plt.title("Training session's progress over iterations")
    # plt.legend(loc='upper right', shadow=True)
    # plt.ylabel('Training Progress (Loss or Accuracy values)')
    # plt.xlabel('Training iteration')
    #
    # plt.show()
    #
    #
    # # Results
    #
    # predictions = one_hot_predictions.argmax(1)
    #
    # print("Testing Accuracy: {}%".format(100*accuracy))
    #
    # print("")
    # print("Precision: {}%".format(100*metrics.precision_score(y_test, predictions, average="weighted")))
    # print("Recall: {}%".format(100*metrics.recall_score(y_test, predictions, average="weighted")))
    # print("f1_score: {}%".format(100*metrics.f1_score(y_test, predictions, average="weighted")))
    #
    # print("")
    # print("Confusion Matrix:")
    # confusion_matrix = metrics.confusion_matrix(y_test, predictions)
    # print(confusion_matrix)
    # normalised_confusion_matrix = np.array(confusion_matrix, dtype=np.float32)/np.sum(confusion_matrix)*100
    #
    # print("")
    # print("Confusion matrix (normalised to % of total test data):")
    # print(normalised_confusion_matrix)
    # print("Note: training and testing data is not equally distributed amongst classes, ")
    # print("so it is normal that more than a 6th of the data is correctly classifier in the last category.")
    #
    # # Plot Results:
    # width = 12
    # height = 12
    # plt.figure(figsize=(width, height))
    # plt.imshow(
    #     normalised_confusion_matrix,
    #     interpolation='nearest',
    #     cmap=plt.cm.rainbow
    # )
    # plt.title("Confusion matrix \n(normalised to % of total test data)")
    # plt.colorbar()
    # tick_marks = np.arange(n_classes)
    # plt.xticks(tick_marks, LABELS, rotation=90)
    # plt.yticks(tick_marks, LABELS)
    # plt.tight_layout()
    # plt.ylabel('True label')
    # plt.xlabel('Predicted label')
    # plt.show()


    sess.close()

def draw_plot(one_hot_predictions, final_loss, accuracy, train_losses, train_accuracies, test_losses, test_accuracies, training_iters,y_test ):
    print("FINAL RESULT: " + \
          "Batch Loss = {}".format(final_loss) + \
          ", Accuracy = {}".format(accuracy))



    font = {
        'family' : 'Bitstream Vera Sans',
        'weight' : 'bold',
        'size'   : 18
    }
    matplotlib.rc('font', **font)

    width = 12
    height = 12
    plt.figure(figsize=(width, height))

    indep_train_axis = np.array(range(batch_size, (len(train_losses)+1)*batch_size, batch_size))
    plt.plot(indep_train_axis, np.array(train_losses),     "b--", label="Train losses")
    plt.plot(indep_train_axis, np.array(train_accuracies), "g--", label="Train accuracies")

    indep_test_axis = np.append(
        np.array(range(batch_size, len(test_losses)*display_iter, display_iter)[:-1]),
        [training_iters]
    )
    plt.plot(indep_test_axis, np.array(test_losses),     "b-", label="Test losses")
    plt.plot(indep_test_axis, np.array(test_accuracies), "g-", label="Test accuracies")

    plt.title("Training session's progress over iterations")
    plt.legend(loc='upper right', shadow=True)
    plt.ylabel('Training Progress (Loss or Accuracy values)')
    plt.xlabel('Training iteration')

    plt.show()


    # Results

    predictions = one_hot_predictions.argmax(1)

    print("Testing Accuracy: {}%".format(100*accuracy))

    print("")
    print("Precision: {}%".format(100*metrics.precision_score(y_test, predictions, average="weighted")))
    print("Recall: {}%".format(100*metrics.recall_score(y_test, predictions, average="weighted")))
    print("f1_score: {}%".format(100*metrics.f1_score(y_test, predictions, average="weighted")))

    print("")
    print("Confusion Matrix:")
    confusion_matrix = metrics.confusion_matrix(y_test, predictions)
    print(confusion_matrix)
    normalised_confusion_matrix = np.array(confusion_matrix, dtype=np.float32)/np.sum(confusion_matrix)*100

    print("")
    print("Confusion matrix (normalised to % of total test data):")
    print(normalised_confusion_matrix)
    print("Note: training and testing data is not equally distributed amongst classes, ")
    print("so it is normal that more than a 6th of the data is correctly classifier in the last category.")

    # Plot Results:
    width = 12
    height = 12
    plt.figure(figsize=(width, height))
    plt.imshow(
        normalised_confusion_matrix,
        interpolation='nearest',
        cmap=plt.cm.rainbow
    )
    plt.title("Confusion matrix \n(normalised to % of total test data)")
    plt.colorbar()
    tick_marks = np.arange(n_classes)
    plt.xticks(tick_marks, LABELS, rotation=90)
    plt.yticks(tick_marks, LABELS)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


if __name__ == "__main__":
    rnn_train()





