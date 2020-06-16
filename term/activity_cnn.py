import os
import numpy as np
from keras.utils.np_utils import to_categorical
import matplotlib.pyplot as plt
from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn import cluster
import sys
import librosa
import six
from keras.models import Model
from keras.layers import (
    Input,
    Activation,
    Dense,
    Flatten
)
from keras.layers.convolutional import (
    Conv2D,
    MaxPooling2D,
    AveragePooling2D
)
from keras.layers.merge import add
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
import glob
from keras import backend as K
from resnet_bk import ResnetBuilder

def _bn_relu(input):
    """Helper to build a BN -> relu block
    """
    norm = BatchNormalization(axis=CHANNEL_AXIS)(input)
    return Activation("relu")(norm)


def _conv_bn_relu(**conv_params):
    """Helper to build a conv -> BN -> relu block
    """
    filters = conv_params["filters"]
    kernel_size = conv_params["kernel_size"]
    strides = conv_params.setdefault("strides", (1, 1))
    kernel_initializer = conv_params.setdefault("kernel_initializer", "he_normal")
    padding = conv_params.setdefault("padding", "same")
    kernel_regularizer = conv_params.setdefault("kernel_regularizer", l2(1.e-4))

    def f(input):
        conv = Conv2D(filters=filters, kernel_size=kernel_size,
                      strides=strides, padding=padding,
                      kernel_initializer=kernel_initializer,
                      kernel_regularizer=kernel_regularizer)(input)
        return _bn_relu(conv)

    return f


def _bn_relu_conv(**conv_params):
    """Helper to build a BN -> relu -> conv block.
    This is an improved scheme proposed in http://arxiv.org/pdf/1603.05027v2.pdf
    """
    filters = conv_params["filters"]
    kernel_size = conv_params["kernel_size"]
    strides = conv_params.setdefault("strides", (1, 1))
    kernel_initializer = conv_params.setdefault("kernel_initializer", "he_normal")
    padding = conv_params.setdefault("padding", "same")
    kernel_regularizer = conv_params.setdefault("kernel_regularizer", l2(1.e-4))

    def f(input):
        activation = _bn_relu(input)
        return Conv2D(filters=filters, kernel_size=kernel_size,
                      strides=strides, padding=padding,
                      kernel_initializer=kernel_initializer,
                      kernel_regularizer=kernel_regularizer)(activation)

    return f


def _shortcut(input, residual):
    """Adds a shortcut between input and residual block and merges them with "sum"
    """
    # Expand channels of shortcut to match residual.
    # Stride appropriately to match residual (width, height)
    # Should be int if network architecture is correctly configured.
    input_shape = K.int_shape(input)
    residual_shape = K.int_shape(residual)
    stride_width = int(round(input_shape[ROW_AXIS] / residual_shape[ROW_AXIS]))
    stride_height = int(round(input_shape[COL_AXIS] / residual_shape[COL_AXIS]))
    equal_channels = input_shape[CHANNEL_AXIS] == residual_shape[CHANNEL_AXIS]

    shortcut = input
    # 1 X 1 conv if shape is different. Else identity.
    if stride_width > 1 or stride_height > 1 or not equal_channels:
        shortcut = Conv2D(filters=residual_shape[CHANNEL_AXIS],
                          kernel_size=(1, 1),
                          strides=(stride_width, stride_height),
                          padding="valid",
                          kernel_initializer="he_normal",
                          kernel_regularizer=l2(0.0001))(input)

    return add([shortcut, residual])


def _residual_block(block_function, filters, repetitions, is_first_layer=False):
    """Builds a residual block with repeating bottleneck blocks.
    """
    def f(input):
        for i in range(repetitions):
            init_strides = (1, 1)
            if i == 0 and not is_first_layer:
                init_strides = (2, 2)
            input = block_function(filters=filters, init_strides=init_strides,
                                   is_first_block_of_first_layer=(is_first_layer and i == 0))(input)
        return input
    return f


def basic_block(filters, init_strides=(1, 1), is_first_block_of_first_layer=False):
    """Basic 3 X 3 convolution blocks for use on resnets with layers <= 34.
    Follows improved proposed scheme in http://arxiv.org/pdf/1603.05027v2.pdf
    """
    def f(input):

        if is_first_block_of_first_layer:
            # don't repeat bn->relu since we just did bn->relu->maxpool
            conv1 = Conv2D(filters=filters, kernel_size=(3, 3),
                           strides=init_strides,
                           padding="same",
                           kernel_initializer="he_normal",
                           kernel_regularizer=l2(1e-4))(input)
        else:
            conv1 = _bn_relu_conv(filters=filters, kernel_size=(3, 3),
                                  strides=init_strides)(input)

        residual = _bn_relu_conv(filters=filters, kernel_size=(3, 3))(conv1)
        return _shortcut(input, residual)

    return f


def bottleneck(filters, init_strides=(1, 1), is_first_block_of_first_layer=False):
    """Bottleneck architecture for > 34 layer resnet.
    Follows improved proposed scheme in http://arxiv.org/pdf/1603.05027v2.pdf
    Returns:
        A final conv layer of filters * 4
    """
    def f(input):

        if is_first_block_of_first_layer:
            # don't repeat bn->relu since we just did bn->relu->maxpool
            conv_1_1 = Conv2D(filters=filters, kernel_size=(1, 1),
                              strides=init_strides,
                              padding="same",
                              kernel_initializer="he_normal",
                              kernel_regularizer=l2(1e-4))(input)
        else:
            conv_1_1 = _bn_relu_conv(filters=filters, kernel_size=(1, 1),
                                     strides=init_strides)(input)

        conv_3_3 = _bn_relu_conv(filters=filters, kernel_size=(3, 3))(conv_1_1)
        residual = _bn_relu_conv(filters=filters * 4, kernel_size=(1, 1))(conv_3_3)
        return _shortcut(input, residual)

    return f


def _handle_dim_ordering():
    global ROW_AXIS
    global COL_AXIS
    global CHANNEL_AXIS
    if K.image_data_format() == 'channels_last':
    #if K.image_dim_ordering() == 'tf':
        ROW_AXIS = 1
        COL_AXIS = 2
        CHANNEL_AXIS = 3
    else:
        CHANNEL_AXIS = 1
        ROW_AXIS = 2
        COL_AXIS = 3


def _get_block(identifier):
    if isinstance(identifier, six.string_types):
        res = globals().get(identifier)
        if not res:
            raise ValueError('Invalid {}'.format(identifier))
        return res
    return identifier


class ResnetBuilder(object):
    @staticmethod
    def build(input_shape, num_outputs, block_fn, repetitions):
        """Builds a custom ResNet like architecture.
        Args:
            input_shape: The input shape in the form (nb_channels, nb_rows, nb_cols)
            num_outputs: The number of outputs at final softmax layer
            block_fn: The block function to use. This is either `basic_block` or `bottleneck`.
                The original paper used basic_block for layers < 50
            repetitions: Number of repetitions of various block units.
                At each block unit, the number of filters are doubled and the input size is halved
        Returns:
            The keras `Model`.
        """
        _handle_dim_ordering()
        if len(input_shape) != 3:
            raise Exception("Input shape should be a tuple (nb_channels, nb_rows, nb_cols)")

        # Permute dimension order if necessary
        if K.image_data_format() == 'channels_first':
        #if K.image_dim_ordering() == 'tf':
            input_shape = (input_shape[1], input_shape[2], input_shape[0])

        # Load function from str if needed.
        block_fn = _get_block(block_fn)

        input = Input(shape=input_shape)
        conv1 = _conv_bn_relu(filters=64, kernel_size=(7, 7), strides=(2, 2))(input)
        pool1 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding="same")(conv1)

        block = pool1
        filters = 64
        for i, r in enumerate(repetitions):
            block = _residual_block(block_fn, filters=filters, repetitions=r, is_first_layer=(i == 0))(block)
            filters *= 2

        # Last activation
        block = _bn_relu(block)

        # Classifier block
        block_shape = K.int_shape(block)
        pool2 = AveragePooling2D(pool_size=(block_shape[ROW_AXIS], block_shape[COL_AXIS]),
                                 strides=(1, 1))(block)
        flatten1 = Flatten()(pool2)
        dense = Dense(units=num_outputs, kernel_initializer="he_normal",
                      activation="softmax")(flatten1)

        model = Model(inputs=input, outputs=dense)
        return model

    @staticmethod
    def build_resnet_18(input_shape, num_outputs):
        return ResnetBuilder.build(input_shape, num_outputs, basic_block, [2, 2, 2, 2])

    @staticmethod
    def build_resnet_34(input_shape, num_outputs):
        return ResnetBuilder.build(input_shape, num_outputs, basic_block, [3, 4, 6, 3])

    @staticmethod
    def build_resnet_50(input_shape, num_outputs):
        return ResnetBuilder.build(input_shape, num_outputs, bottleneck, [3, 4, 6, 3])

    @staticmethod
    def build_resnet_101(input_shape, num_outputs):
        return ResnetBuilder.build(input_shape, num_outputs, bottleneck, [3, 4, 23, 3])

    @staticmethod
    def build_resnet_152(input_shape, num_outputs):
        return ResnetBuilder.build(input_shape, num_outputs, bottleneck, [3, 8, 36, 3])


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
# Init param with VQ
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
    freq_axis = np.linspace(0,fs,n_frames)
    return stft, freq_axis


def get_log_mel_transform(p_data, type = None):
    # nfft = 512
    # window_len = 0.4#0.3#0.5  # 1.0#0.3
    # hop_len = 0.1#0.1  # 0.5
    # sr = 50
    # win_type = 'hann'
    # n_coeff = 8
    nfft = 128
    window_len = 1.3  # 0.3#0.5  # 1.0#0.3
    hop_len = 0.3  # 0.1  # 0.5
    sr = 50
    win_type = 'hann'
    n_coeff = 48

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
    """
        select data by label
    """
    tr_data_by_label = tr_data
    te_data_by_label = te_data
    if label is not None:
        tr_data_by_label = tr_data[np.where(tr_data[:,0]==str(label))]
        te_data_by_label = te_data[np.where(te_data[:,0]==str(label))]

    if 'flatten' in type:
        train_x = [x.flatten() for x in tr_data_by_label[:,3]]
        train_y = tr_data_by_label[:,0:1].astype(int)-1
        test_x = [x.flatten() for x in te_data_by_label[:,3]]
        test_y = te_data_by_label[:,0:1].astype(int)-1
    elif 'raw' in type:
        train_x = np.array([x for x in tr_data_by_label[:,3]])
        train_y = tr_data_by_label[:,0:1].astype(int)-1
        test_x =  np.array([x for x in te_data_by_label[:,3]])
        test_y = te_data_by_label[:,0:1].astype(int)-1
    elif 'fft' == type:
        train_x_s = np.array([get_log_mel_transform(x) for x in tr_data_by_label[:,3]])
        train_x = np.squeeze(train_x_s)
        train_y = tr_data_by_label[:,0:1].astype(int)-1
        test_x_s =  np.array([get_log_mel_transform(x) for x in te_data_by_label[:,3]])
        test_x = np.squeeze(test_x_s)
        test_y = te_data_by_label[:,0:1].astype(int)-1
        del train_x_s
        del test_x_s
    elif 'nnfft' == type:
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
        make init param centroid point for vq
    """

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
    if type == "NN":
        feature = np.vstack((vq_feature1, vq_feature2, vq_feature3, vq_feature4, vq_feature5, vq_feature6)).T
    else:
        feature = np.concatenate((vq_feature1, vq_feature2, vq_feature3, vq_feature4, vq_feature5, vq_feature6))
    return feature


def get_vg_train(p_data):
    """
        make init centroid point for vq
    """
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
    print(all_centroid_init)


def load_datasets(data_path, npz=None , shuffle= False):
    """Load dataset unsing numpy wih data path
        . Datasets :
          . filename : 0009_01_5.txt
            “aaaa_bb_c.txt,” where aaaa is the file ID, bb is the user ID, and c is the ground truth activity class
          . data : The six columns represent accelerations (in standard gravity unit g=9.8m/s2) in X, Y, and Z
                   directions, and angular velocities (in rad/sec) in X, Y, and Z directions
    """

    if npz == None:
        cnt = 0
        #np_all_data = np.empty(shape=[0,9])
        rowlist = list()
        for file in glob.glob(data_path + "/*.txt"):
            col_list = list()
            cnt += 1
            row = np.loadtxt(file)
            file_index, user_id, activity_class = os.path.basename(os.path.splitext(file)[0]).split("_")
            col_list.append(str(activity_class))
            col_list.append(int(user_id))
            col_list.append(int(file_index))
            col_list.append(row)
            rowlist.append(col_list)
            del col_list
            if cnt % 100 == 0:
                print(cnt)
        np_all_data = np.array(rowlist)
        np.savez("./npz/activity_nparray3.npz", data = np_all_data)
        print("save complete")
    else:
        np_all_data = np.load("./npz/activity_nparray3.npz",allow_pickle=True)['data']
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


def _test_model_compile(model):
    K.set_image_data_format('channels_last')
    model.compile(loss="categorical_crossentropy", optimizer="sgd")
    return model


def test_resnet18(input_shape, n_classes):
    model = ResnetBuilder.build_resnet_18((input_shape[0], input_shape[1], input_shape[2]), n_classes)
    model.compile(optimizer=optimizer, loss=objective, metrics=['accuracy'])
    return model


def test_resnet34(input_shape, n_classes):
    model = ResnetBuilder.build_resnet_34((input_shape[0], input_shape[1], input_shape[2]), n_classes)
    model.compile(optimizer=optimizer, loss=objective, metrics=['accuracy'])
    return model


def test_resnet50(input_shape, n_classes):
    model = ResnetBuilder.build_resnet_50((input_shape[0], input_shape[1], input_shape[2]), n_classes)
    model.compile(optimizer=optimizer, loss=objective, metrics=['accuracy'])
    return model


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


def run_resNet(input_shape):
    history = LossHistory()
    # x_train & x_test reshaped because of single channel !
    model.fit(X_train.reshape((-1, input_shape[0], input_shape[1], input_shape[2])), y_train_onehot, batch_size=batch_size, epochs=nb_epoch,
              validation_data=(X_test, y_test_onehot),#validation_split=0.25,
              verbose=1, shuffle=True, callbacks=[history, early_stopping, reduce_lr,mcp_save])

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


def draw_plot(one_hot_predictions, y_tes ):

    predictions = one_hot_predictions.argmax(1)

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
    plt.savefig('resnet_cnn_50.png')
    plt.show()

if __name__ == "__main__":
    os.makedirs("./save/", exist_ok=True)
    #data_path = "./mlpr20_project_train_data"
    data_path = "./new_dataset"
    tr_data, te_data = load_datasets(data_path, False, shuffle=True)
    X_train, y_train, X_test, y_test = load_datasets_by_label(tr_data, te_data, label=None, type='nnfft')#nnvq, nnfft, raw

    X_train = np.expand_dims(X_train, axis=3)
    X_test =  np.expand_dims(X_test, axis=3)

    y_train_onehot = to_categorical(y_train, num_classes=None, dtype='float32')
    y_test_onehot = to_categorical(y_test, num_classes=None, dtype='float32')

    print("Train Set Size = {} images".format(y_train.shape[0]))
    print("Test Set Size = {} images".format(y_test.shape[0]))

    input_shape = (X_train.shape[1],X_train.shape[2],X_train.shape[3])
    print("Labels : {}".format(y_train[0:5]))

    optimizer = 'adam'
    objective = 'categorical_crossentropy'

    model_type = "resnet18" #resnet34, resnet50

    if "resnet18" == model_type:
        model = test_resnet18(input_shape,n_classes)
    elif "resnet34" == model_type:
        model = test_resnet34(input_shape,n_classes)
    elif "resnet50" == model_type:
        model = test_resnet50(input_shape,n_classes)
    model.summary()
    nb_epoch = 50
    batch_size = 128

    early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='auto')
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1,
                                  patience=5, min_lr=0.000001, verbose=1)
    mcp_save = ModelCheckpoint('./save/'+model_type+'.hdf5', save_best_only=True, monitor='val_loss', mode='min', verbose = 1)

    predictions, history = run_resNet(input_shape)
    draw_plot(predictions, y_test)
