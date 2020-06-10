
import os
import numpy as np
from keras import layers
from keras.datasets import mnist
from keras.utils.np_utils import to_categorical
import matplotlib.pyplot as plt
from keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D
from keras.layers import AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D
from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping
from keras.models import Model
from keras.initializers import glorot_uniform
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn import cluster
import sys
import librosa



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
    if type != "NN":
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
    elif 'nnftt' in type:
        #train_x = tr_data_by_label[:,3]
        train_x_s = np.array([get_log_mel_transform(x,"NN") for x in tr_data_by_label[:,3]])
        train_x = np.squeeze(train_x_s)
        train_y = tr_data_by_label[:,0:1].astype(int)-1
        test_x_s =  np.array([get_log_mel_transform(x, "NN") for x in te_data_by_label[:,3]])
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
    elif 'nnvq' == type:
        train_x_s = np.array([get_vq(x,"NN") for x in tr_data_by_label[:, 3]])
        train_x = np.squeeze(train_x_s)
        train_y = tr_data_by_label[:,0:1].astype(int)-1
        test_x_s =  np.array([get_vq(x,"NN") for x in te_data_by_label[:,3]])
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

    if type == "NN":
        feature = np.vstack((vq_feature1, vq_feature2, vq_feature3, vq_feature4, vq_feature5, vq_feature6)).T
    else:
        feature = np.concatenate((vq_feature1, vq_feature2, vq_feature3, vq_feature4, vq_feature5, vq_feature6))


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

data_path = "./mlpr20_project_train_data"
tr_data, te_data = load_datasets(data_path, False, shuffle=True)
X_train, y_train, X_test, y_test = load_datasets_by_label(tr_data, te_data, label=None, type='nnvq')
#x_train, y_train, x_test, y_test = load_datasets_by_label(tr_data, te_data,  label=None, type='fft')
X_train = np.expand_dims(X_train, axis=3)
X_test =  np.expand_dims(X_test, axis=3)
#(x_train, y_train), (x_test, y_test) = mnist.load_data()
y_train_onehot = to_categorical(y_train, num_classes=None, dtype='float32')
y_test_onehot = to_categorical(y_test, num_classes=None, dtype='float32')

print("Train Set Size = {} images".format(y_train.shape[0]))
print("Test Set Size = {} images".format(y_test.shape[0]))



input_shape = (X_train.shape[1],X_train.shape[2],X_train.shape[3])
# fig1 = plt.figure(figsize = (15,15))
#
# for i in range(5):
#     ax1 = fig1.add_subplot(1,5,i+1)
#     ax1.imshow(x_train[i], interpolation='none', cmap=plt.cm.gray)
#     ax2 = fig1.add_subplot(2,5,i+6)
#     ax2.imshow(x_train[i+6], interpolation='none', cmap=plt.cm.gray)
# plt.show()

print("Labels : {}".format(y_train[0:5]))
#print("Labels : {}".format(y_train[6:11]))

optimizer = 'adam'
objective = 'categorical_crossentropy'


def identity_block(X, f, filters, stage, block):
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    F1, F2, F3 = filters

    X_shortcut = X

    X = Conv2D(filters=F1, kernel_size=(1, 1), strides=(1, 1), padding='valid', name=conv_name_base + '2a',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2a')(X)
    X = Activation('relu')(X)

    X = Conv2D(filters=F2, kernel_size=(f, f), strides=(1, 1), padding='same', name=conv_name_base + '2b',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2b')(X)
    X = Activation('relu')(X)

    X = Conv2D(filters=F3, kernel_size=(1, 1), strides=(1, 1), padding='valid', name=conv_name_base + '2c',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2c')(X)

    X = layers.Add()([X, X_shortcut])
    X = Activation('relu')(X)

    return X


def convolutional_block(X, f, filters, stage, block, s=2):
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    F1, F2, F3 = filters

    X_shortcut = X

    X = Conv2D(filters=F1, kernel_size=(1, 1), strides=(s, s), name=conv_name_base + '2a',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2a')(X)
    X = Activation('relu')(X)

    X = Conv2D(filters=F2, kernel_size=(f, f), strides=(1, 1), padding='same', name=conv_name_base + '2b',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2b')(X)
    X = Activation('relu')(X)

    X = Conv2D(filters=F3, kernel_size=(1, 1), strides=(1, 1), name=conv_name_base + '2c',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2c')(X)

    X_shortcut = Conv2D(F3, kernel_size=(1, 1), strides=(s, s), name=conv_name_base + '1',
                        kernel_initializer=glorot_uniform(seed=0))(X_shortcut)
    X_shortcut = BatchNormalization(axis=3, name=bn_name_base + '1')(X_shortcut)

    X = layers.Add()([X, X_shortcut])
    X = Activation('relu')(X)

    return X


def resNet(input_shape):
    # Define the input as a tensor with shape input_shape
    X_input = Input((input_shape))

    # Zero-Padding
    X = ZeroPadding2D((3, 3))(X_input)

    # Stage 1
    X = Conv2D(64, (7, 7), strides=(2, 2), name='conv1', )(X)
    X = BatchNormalization(axis=3, name='bn_conv1')(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((3, 3), strides=(2, 2))(X)

    # Stage 2
    X = convolutional_block(X, f=3, filters=[64, 64, 256], stage=2, block='a', s=1)
    X = identity_block(X, 3, [64, 64, 256], stage=2, block='b')
    X = identity_block(X, 3, [64, 64, 256], stage=2, block='c')

    # Stage 3
    X = convolutional_block(X, f=3, filters=[128, 128, 512], stage=3, block='a', s=2)
    X = identity_block(X, 3, [128, 128, 512], stage=3, block='b')
    X = identity_block(X, 3, [128, 128, 512], stage=3, block='c')
    X = identity_block(X, 3, [128, 128, 512], stage=3, block='d')

    # Stage 4
    X = convolutional_block(X, f=3, filters=[256, 256, 1024], stage=4, block='a', s=2)
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='b')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='c')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='d')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='e')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='f')

    # Stage 5
    X = convolutional_block(X, f=3, filters=[512, 512, 2048], stage=5, block='a', s=2)
    X = identity_block(X, 3, [512, 512, 2048], stage=5, block='b')
    X = identity_block(X, 3, [512, 512, 2048], stage=5, block='c')

    # AVGPOOL
    # X = AveragePooling2D((2,2), name='avg_pool')(X)

    # output layer
    X = Flatten()(X)
    X = Dense(y_train_onehot.shape[1], activation='softmax', name='fc' + str(y_train_onehot.shape[1]))(X)

    # Create model
    model = Model(inputs=X_input, outputs=X, name='ResNet50')

    # Compile the model
    model.compile(optimizer=optimizer, loss=objective, metrics=['accuracy'])

    return model


model = resNet(input_shape)

nb_epoch = 1
batch_size = 128


## Callback for loss logging per epoch
class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        self.val_losses = []
        self.acc = []
        self.val_acc = []

    def on_epoch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        self.acc.append(logs.get('acc'))
        self.val_acc.append(logs.get('val_acc'))


early_stopping = EarlyStopping(monitor='val_loss', patience=100, verbose=1, mode='auto')


def run_resNet(input_shape):
    history = LossHistory()
    # x_train & x_test reshaped because of single channel !
    model.fit(X_train.reshape((-1, input_shape[0], input_shape[1], input_shape[2])), y_train_onehot, batch_size=batch_size, epochs=nb_epoch,
              validation_data=(X_test, y_test_onehot),#validation_split=0.25,
              verbose=1, shuffle=True, callbacks=[history, early_stopping])

    predictions = model.predict(X_test.reshape((-1,  input_shape[0], input_shape[1], input_shape[2])), verbose=0)
    return predictions, history


def test_accuracy(predictions):
    err = []
    t = 0
    for i in range(predictions.shape[0]):
        if (np.argmax(predictions[i]) == y_test[i]):
            t = t + 1
        else:
            err.append(i)
    return t, float(t) * 100 / predictions.shape[0], err



def draw_plot(one_hot_predictions,y_test ):


    font = {
        'family' : 'Bitstream Vera Sans',
        'weight' : 'bold',
        'size'   : 18
    }




    predictions = one_hot_predictions.argmax(1)

    #print("Testing Accuracy: {}%".format(100*accuracy))

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
    #plt.colorbar()
    tick_marks = np.arange(n_classes)
    plt.xticks(tick_marks, LABELS, rotation=90)
    plt.yticks(tick_marks, LABELS)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

predictions, history = run_resNet(input_shape)

draw_plot(predictions, y_test)