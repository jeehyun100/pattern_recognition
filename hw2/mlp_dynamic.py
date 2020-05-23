import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Activation
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import PReLU
from tensorflow.keras.layers import ELU
from tensorflow.keras.initializers import TruncatedNormal
from tensorflow.keras.initializers import glorot_normal
from tensorflow.keras.initializers import glorot_uniform


def load_datasets(data_path, dataset_name, one_hot=False):
    """Load dataset unsing numpy wih data path
        . Datasets : p1_train(input,target), p1_test(input,target), p2_train(input,target), p2_test(input,target)

    Args:
        data_path (string) : Path data load
        dataset_name : For seleced dataset name  ['p1', 'p2']

    Returns:
        2-D Array: P1 Train data
        2-D Array: P1 Test data
        2-D Array: P2 Train data
        2-D Array: P2 Test data

    """
    # if 'p1' in data_type:
    np_p1_tr_i = np.loadtxt(data_path + "/p1_train_input.txt")
    np_p1_tr_t = np.loadtxt(data_path + "/p1_train_target.txt")
    np_p1_te_i = np.loadtxt(data_path + "/p1_test_input.txt")
    np_p1_te_t = np.loadtxt(data_path + "/p1_test_target.txt")

    # elif 'p2' in data_type:
    np_p2_tr_i = np.loadtxt(data_path + "/p2_train_input.txt")
    np_p2_tr_t = np.loadtxt(data_path + "/p2_train_target.txt")
    np_p2_te_i = np.loadtxt(data_path + "/p2_test_input.txt")
    np_p2_te_t = np.loadtxt(data_path + "/p2_test_target.txt")

    if dataset_name == 'p1':
        # Concatenate train value and target
        np_p_tr_a = np.column_stack([np_p1_tr_i, np_p1_tr_t])
        np_p_te_a = np.column_stack([np_p1_te_i, np_p1_te_t])

    elif dataset_name == 'p2':
        # Concatenate train value and target
        np_p_tr_a = np.column_stack([np_p2_tr_i, np_p2_tr_t])
        np_p_te_a = np.column_stack([np_p2_te_i, np_p2_te_t])
    elif dataset_name == 'all':
        # Concatenate train value and target
        np_p1_tr_a = np.column_stack([np_p1_tr_i, np_p1_tr_t])
        np_p1_te_a = np.column_stack([np_p1_te_i, np_p1_te_t])
        np_p2_tr_a = np.column_stack([np_p2_tr_i, np_p2_tr_t])
        np_p2_te_a = np.column_stack([np_p2_te_i, np_p2_te_t])
        np_p_tr_a = np.vstack([np_p1_tr_a, np_p2_tr_a])
        np_p_te_a = np.vstack([np_p1_te_a, np_p2_te_a])

    tr_target = np_p_tr_a[:, 2]
    te_target = np_p_te_a[:, 2]

    if one_hot:
        tr_target = tr_target.reshape(-1, 1)
        te_target = te_target.reshape(-1, 1)
        enc_tr = OneHotEncoder()
        enc_tr.fit(tr_target)
        tr_target = enc_tr.transform(tr_target).toarray()

        enc_te = OneHotEncoder()
        enc_te.fit(te_target)
        te_target = enc_tr.transform(te_target).toarray()

    return np_p_tr_a[:, 0:2], tr_target, np_p_te_a[:, 0:2], te_target


def custom_mlp(input_shape, depth, af='relu', num_classes=2):
    """ResNet Version 2 Model builder [b]

    # Arguments
        input_shape (tensor): shape of input image tensor
        depth (int): number of core convolutional layers
        num_classes (int): number of classes (CIFAR10 has 10)
    # Returns
        model (Model): Keras model instance
    """
    model = None
    x = None
    outputs = None
    inputs = None

    inputs = tf.keras.Input(shape=input_shape)
    if af == 'LeakyReLU':
        x = Dense(10)(inputs)
        x = Activation(LeakyReLU(alpha=0.1))(x)
    elif af == 'PReLU':
        x = Dense(10)(inputs)
        x = Activation(PReLU())(x)
    elif af == 'ELU':
        x = Dense(10)(inputs)
        x = Activation(ELU(alpha=0.1))(x)
    else:
        x = Dense(10, activation=af)(inputs)

    for _ in range(depth):
        if af == 'LeakyReLU':
            x = Dense(10)(x)
            x = Activation(LeakyReLU(alpha=0.1))(x)
        elif af == 'PReLU':
            x = Dense(10)(x)
            x = Activation(PReLU())(x)
        elif af == 'ELU':
            x = Dense(10)(x)
            x = Activation(ELU(alpha=0.1))(x)
        else:
            x = Dense(10, activation=af)(x)

    outputs = Dense(num_classes,
                    activation='softmax',
                    kernel_initializer='he_normal')(x)

    # Instantiate model.
    model = Model(inputs=inputs, outputs=outputs)
    return model


def custom_mlp_with_init(input_shape, depth, af='relu', num_classes=2, init_type='None'):
    """custom_mlp_with_init

    # Returns
        model (Model): Keras model instance
    """
    model = None
    x = None
    outputs = None
    inputs = None

    inputs = tf.keras.Input(shape=input_shape)

    if af == 'LeakyReLU':
        if init_type == 'xia':
            initializer = glorot_uniform()
        else:
            initializer = TruncatedNormal(mean=0.0, stddev=0.5, seed=1)
        # x = Dense(10, kernel_initializer='glorot_uniform', bias_initializer='zeros')(inputs)
        x = Dense(10, kernel_initializer=initializer, bias_initializer='zeros')(inputs)
        x = Activation(LeakyReLU(alpha=0.1))(x)
    elif af == 'PReLU':
        if init_type == 'xia':
            initializer = glorot_uniform()
        else:
            initializer = TruncatedNormal(mean=0.0, stddev=0.5, seed=1)
        # x = Dense(10, kernel_initializer='glorot_uniform', bias_initializer='zeros')(inputs)
        x = Dense(10, kernel_initializer=initializer, bias_initializer='zeros')(inputs)
        x = Activation(PReLU())(x)
    elif af == 'ELU':
        if init_type == 'xia':
            initializer = glorot_uniform()
        else:
            initializer = TruncatedNormal(mean=0.0, stddev=0.5, seed=1)
        # initializer = TruncatedNormal(mean=0.0, stddev=0.5, seed=1)
        # x = Dense(10, kernel_initializer='glorot_uniform', bias_initializer='zeros')(inputs)
        x = Dense(10, kernel_initializer=initializer, bias_initializer='zeros')(inputs)
        x = Activation(ELU(alpha=0.1))(x)
    else:
        if init_type == 'xia':
            initializer = glorot_uniform()
        else:
            initializer = TruncatedNormal(mean=0.0, stddev=0.5, seed=1)
        # x = Dense(10, kernel_initializer='glorot_uniform', bias_initializer='zeros', activation=af)(inputs)
        x = Dense(10, kernel_initializer=initializer, bias_initializer='zeros', activation=af)(inputs)

    for _ in range(depth):
        if af == 'LeakyReLU':
            if init_type == 'xia':
                initializer = glorot_uniform()
            else:
                initializer = TruncatedNormal(mean=0.0, stddev=0.5, seed=1)
            # x = Dense(10, kernel_initializer='glorot_uniform', bias_initializer='zeros')(x)
            x = Dense(10, kernel_initializer=initializer, bias_initializer='zeros')(x)
            x = Activation(LeakyReLU(alpha=0.1))(x)
        elif af == 'PReLU':
            if init_type == 'xia':
                initializer = glorot_uniform()
            else:
                initializer = TruncatedNormal(mean=0.0, stddev=0.5, seed=1)
            # x = Dense(10, kernel_initializer='glorot_uniform', bias_initializer='zeros')(x)
            x = Dense(10, kernel_initializer=initializer, bias_initializer='zeros')(x)
            x = Activation(PReLU())(x)
        elif af == 'ELU':
            if init_type == 'xia':
                initializer = glorot_uniform()
            else:
                initializer = TruncatedNormal(mean=0.0, stddev=0.5, seed=1)
            # x = Dense(10, kernel_initializer='glorot_uniform', bias_initializer='zeros')(x)
            x = Dense(10, kernel_initializer=initializer, bias_initializer='zeros')(x)
            x = Activation(ELU(alpha=0.1))(x)
        else:
            if init_type == 'xia':
                initializer = glorot_uniform()
            else:
                initializer = TruncatedNormal(mean=0.0, stddev=0.5, seed=1)
            # x = Dense(10, kernel_initializer='glorot_uniform', bias_initializer='zeros', activation=af)(x)
            x = Dense(10, kernel_initializer=initializer, bias_initializer='zeros', activation=af)(x)

    outputs = Dense(num_classes,
                    activation='softmax',
                    kernel_initializer='he_normal')(x)

    # Instantiate model.
    model = Model(inputs=inputs, outputs=outputs)
    return model


def custom_mlp_best(input_shape, depth, af='relu', num_classes=2, init_type='None'):
    """ Make custom mlp

    # Returns
        model (Model): Keras model instance
    """
    model = None
    x = None
    outputs = None
    inputs = None

    inputs = tf.keras.Input(shape=input_shape)
    # v2 performs Conv2D with BN-ReLU on input before splitting into 2 paths
    initializer = glorot_uniform()

    x = Dense(256, kernel_initializer='he_normal')(inputs)
    x = BatchNormalization()(x)
    x = Activation(PReLU())(x)

    x = Dense(1024, kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = Activation(PReLU())(x)
    #
    x = Dense(1024, kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = Activation(PReLU())(x)
    x = Dense(5, kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = Activation(PReLU())(x)

    outputs = Dense(num_classes,
                    activation='softmax',
                    kernel_initializer='he_normal')(x)

    # Instantiate model.
    model = Model(inputs=inputs, outputs=outputs)
    return model


def custom_mlp_with_mse(input_shape, depth, af='relu', num_classes=1, init_type='None'):
    """ mlp model with mse loss
    # Returns
        model (Model): Keras model instance
    """
    model = None
    x = None
    outputs = None
    inputs = None

    inputs = tf.keras.Input(shape=input_shape)

    if af == 'LeakyReLU':
        if init_type == 'xia':
            initializer = glorot_uniform()
        else:
            initializer = TruncatedNormal(mean=0.0, stddev=0.5, seed=1)
        # x = Dense(10, kernel_initializer='glorot_uniform', bias_initializer='zeros')(inputs)
        x = Dense(10, kernel_initializer=initializer, bias_initializer='zeros')(inputs)
        x = Activation(LeakyReLU(alpha=0.1))(x)
    elif af == 'PReLU':
        if init_type == 'xia':
            initializer = glorot_uniform()
        else:
            initializer = TruncatedNormal(mean=0.0, stddev=0.5, seed=1)
        # x = Dense(10, kernel_initializer='glorot_uniform', bias_initializer='zeros')(inputs)
        x = Dense(10, kernel_initializer=initializer, bias_initializer='zeros')(inputs)
        x = Activation(PReLU())(x)
    elif af == 'ELU':
        if init_type == 'xia':
            initializer = glorot_uniform()
        else:
            initializer = TruncatedNormal(mean=0.0, stddev=0.5, seed=1)
        # initializer = TruncatedNormal(mean=0.0, stddev=0.5, seed=1)
        # x = Dense(10, kernel_initializer='glorot_uniform', bias_initializer='zeros')(inputs)
        x = Dense(10, kernel_initializer=initializer, bias_initializer='zeros')(inputs)
        x = Activation(ELU(alpha=0.1))(x)
    else:
        if init_type == 'xia':
            initializer = glorot_uniform()
        else:
            initializer = TruncatedNormal(mean=0.0, stddev=0.5, seed=1)
        # x = Dense(10, kernel_initializer='glorot_uniform', bias_initializer='zeros', activation=af)(inputs)
        x = Dense(10, kernel_initializer=initializer, bias_initializer='zeros', activation=af)(inputs)

    for _ in range(depth):
        if af == 'LeakyReLU':
            if init_type == 'xia':
                initializer = glorot_uniform()
            else:
                initializer = TruncatedNormal(mean=0.0, stddev=0.5, seed=1)
            # x = Dense(10, kernel_initializer='glorot_uniform', bias_initializer='zeros')(x)
            x = Dense(10, kernel_initializer=initializer, bias_initializer='zeros')(x)
            x = Activation(LeakyReLU(alpha=0.1))(x)
        elif af == 'PReLU':
            if init_type == 'xia':
                initializer = glorot_uniform()
            else:
                initializer = TruncatedNormal(mean=0.0, stddev=0.5, seed=1)
            # x = Dense(10, kernel_initializer='glorot_uniform', bias_initializer='zeros')(x)
            x = Dense(10, kernel_initializer=initializer, bias_initializer='zeros')(x)
            x = Activation(PReLU())(x)
        elif af == 'ELU':
            if init_type == 'xia':
                initializer = glorot_uniform()
            else:
                initializer = TruncatedNormal(mean=0.0, stddev=0.5, seed=1)
            # x = Dense(10, kernel_initializer='glorot_uniform', bias_initializer='zeros')(x)
            x = Dense(10, kernel_initializer=initializer, bias_initializer='zeros')(x)
            x = Activation(ELU(alpha=0.1))(x)
        else:
            if init_type == 'xia':
                initializer = glorot_uniform()
            else:
                initializer = TruncatedNormal(mean=0.0, stddev=0.5, seed=1)
            # x = Dense(10, kernel_initializer='glorot_uniform', bias_initializer='zeros', activation=af)(x)
            x = Dense(10, kernel_initializer=initializer, bias_initializer='zeros', activation=af)(x)

    outputs = Dense(num_classes,
                    activation='sigmoid',
                    kernel_initializer='he_normal')(x)

    # Instantiate model.
    model = Model(inputs=inputs, outputs=outputs)
    return model


def train_1(data_path, datasets, name='mlp', depth=10):
    """
    mlp model hidden layer change
    """
    result_plots = list()
    for ds in datasets:
        result_plots.clear()
        for i in range(depth):
            model_name, plot = mlp_exam(data_path, ds, name, i, )
            result_plots.append([model_name, plot])
        plot_history_loss(result_plots)
        plot_history_acc(result_plots)


def train_2(data_path, datasets, name='mlp', depth=10):
    """
    mlp model activation chagne
    """
    result_plots = list()
    activation_fn = ['sigmoid', 'tanh', 'relu', 'LeakyReLU', 'PReLU', 'ELU']
    # activation_fn = ['sigmoid', 'ELU']
    for ds in datasets:
        result_plots.clear()
        for i in range(depth)[5:6]:
            for af in activation_fn:
                model_name, plot = mlp_exam(data_path, ds, name, i, af=af, init=True, init_type='None')
                result_plots.append([model_name, plot])

        plot_history_loss(result_plots)
        plot_history_acc(result_plots)


def train_2_1(data_path, datasets, name='mlp', depth=10):
    """
    mlp model sigmoid and mse test
    """
    result_plots = list()
    activation_fn = ['sigmoid', 'tanh', 'relu', 'LeakyReLU', 'PReLU', 'ELU']
    # activation_fn = ['sigmoid', 'ELU']
    for ds in datasets:
        result_plots.clear()
        for i in range(depth)[5:6]:
            for af in activation_fn:
                model_name, plot = mlp_exam_mse(data_path, ds, name, i, af=af, init=True, init_type='xia')
                result_plots.append([model_name, plot])

        plot_history_mse(result_plots)


def train_3(data_path, datasets, name='mlp', depth=10):
    """
    mlp model optimizer change
    """
    result_plots = list()
    optimizers = ['SGD', 'RMSprop', 'Adam', 'Nadam']
    for ds in datasets:

        for i in list(map(list(range(depth)).__getitem__, (1, 2, 7))):  # range(depth)[1,2,7]:
            result_plots.clear()
            for opti in optimizers:
                model_name, plot = mlp_exam_optimizer(data_path, ds, name, i, af='relu', init=True, init_type='xia',
                                                      optimizer=opti)
                result_plots.append([model_name, plot])
            plot_history_optimizer(result_plots)


def train_best(data_path, datasets, name='mlp', depth=10):
    result_plots = list()

    for ds in datasets:
        result_plots.clear()

        model_name, plot = mlp_best(data_path, ds, name, 1, af='relu', init=True, init_type='None')
        result_plots.append([model_name, plot])
        plot_history_loss(result_plots)
        plot_history_acc(result_plots)


def mlp_exam(data_path, ds, name='mlp', depth=0, af='relu', init=False, init_type='None'):
    model_name = name
    data_name = ds
    # depth = 10
    model_name = name + "_" + str(depth) + "_" + data_name + "_" + af
    tr_data, tr_target, te_data, te_target = load_datasets(data_path, data_name, True)
    input_shape = 2
    if init:
        model = custom_mlp_with_init(input_shape=input_shape, depth=depth, af=af, init_type=init_type)
    else:
        model = custom_mlp(input_shape=input_shape, depth=depth, af=af)

    optimizer = tf.keras.optimizers.RMSprop(0.001)
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy', 'categorical_crossentropy'])
    print(model.summary())

    ckpt_model_name = "./ckpt/" + model_name  # + '_{epoch:02d}-{val_loss:.4f}'
    csv_name = "./csv/" + model_name
    earlyStopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='min')
    mcp_save = ModelCheckpoint(ckpt_model_name + ".h5", save_best_only=True,
                               monitor='val_loss', mode='min')
    reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=7, verbose=1, epsilon=1e-4, mode='min')

    if init:
        bigger_history = model.fit(tr_data, tr_target,
                                   epochs=300,
                                   batch_size=40,
                                   validation_data=(te_data, te_target),
                                   callbacks=[mcp_save],
                                   verbose=2)
    else:
        bigger_history = model.fit(tr_data, tr_target,
                                   epochs=1000,
                                   batch_size=40,
                                   validation_data=(te_data, te_target),
                                   callbacks=[earlyStopping, mcp_save, reduce_lr_loss],
                                   verbose=2)

    hist = pd.DataFrame(bigger_history.history)
    hist['epoch'] = bigger_history.epoch
    hist.to_csv(csv_name + ".csv")
    return model_name, hist


def mlp_best(data_path, ds, name='mlp', depth=0, af='relu', init=False, init_type='None'):
    model_name = name
    data_name = ds
    # depth = 10
    model_name = name + "_" + str(depth) + "_" + data_name + "_" + af
    tr_data, tr_target, te_data, te_target = load_datasets(data_path, data_name, True)
    input_shape = 2

    model = custom_mlp_best(input_shape=input_shape, depth=depth, af=af, init_type=init_type)
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy', 'categorical_crossentropy'])
    print(model.summary())

    ckpt_model_name = "./ckpt/" + model_name
    csv_name = "./csv/" + model_name
    earlyStopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                     patience=40, verbose=0, mode='min')
    mcp_save = ModelCheckpoint(ckpt_model_name + ".h5", save_best_only=True,
                               monitor='val_loss', mode='min')
    reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.1,
                                       patience=35, verbose=2, epsilon=0.1, mode='min')

    bigger_history = model.fit(tr_data, tr_target,
                               epochs=10000,
                               batch_size=110,
                               validation_data=(te_data, te_target),
                               callbacks=[earlyStopping, mcp_save, reduce_lr_loss],
                               verbose=2)

    hist = pd.DataFrame(bigger_history.history)
    hist['epoch'] = bigger_history.epoch
    hist.to_csv(csv_name + ".csv")

    score = model.evaluate(te_data, te_target)
    print("best model score {0}".format(score))
    return model_name, hist


def mlp_exam_optimizer(data_path, ds, name='mlp', depth=0, af='relu', init=False, init_type='None', optimizer='adam'):
    model_name = name
    data_name = ds
    # depth = 10
    model_name = name + "_" + str(depth) + "_" + data_name + "_" + optimizer
    tr_data, tr_target, te_data, te_target = load_datasets(data_path, data_name, True)
    input_shape = 2

    if init:
        model = custom_mlp_with_init(input_shape=input_shape, depth=depth, af=af, init_type=init_type)
    else:
        model = custom_mlp(input_shape=input_shape, depth=depth, af=af)

    optimizer = tf.keras.optimizers.RMSprop(0.001)
    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy', 'categorical_crossentropy'])
    print(model.summary())

    ckpt_model_name = "./ckpt/" + model_name + '_{epoch:02d}-{val_loss:.4f}'
    csv_name = "./csv/" + model_name
    earlyStopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=20, verbose=0, mode='min')
    mcp_save = ModelCheckpoint(ckpt_model_name + ".h5", save_best_only=True, monitor='val_loss', mode='min')
    reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.99, patience=1000, verbose=1, epsilon=1e-4,
                                       mode='min')

    bigger_history = model.fit(tr_data, tr_target,
                               epochs=1000,
                               batch_size=40,
                               validation_data=(te_data, te_target),
                               callbacks=[earlyStopping, mcp_save, reduce_lr_loss],
                               verbose=2)

    hist = pd.DataFrame(bigger_history.history)
    hist['epoch'] = bigger_history.epoch

    hist.to_csv(csv_name + ".csv")
    return model_name, hist


def mlp_exam_mse(data_path, ds, name='mlp', depth=0, af='relu', init=False, init_type='None'):
    model_name = name
    data_name = ds
    # depth = 10
    model_name = name + "_" + str(depth) + "_" + data_name + "_" + af
    tr_data, tr_target, te_data, te_target = load_datasets(data_path, data_name)
    input_shape = 2
    if init:
        model = custom_mlp_with_mse(input_shape=input_shape, depth=depth, af=af, init_type=init_type)
    else:
        model = custom_mlp(input_shape=input_shape, depth=depth, af=af)

    optimizer = tf.keras.optimizers.RMSprop(0.001)
    model.compile(optimizer='adam',
                  loss='mse',
                  metrics=['mae', 'mse'])
    print(model.summary())

    # model_path = MODEL_SAVE_FOLDER_PATH +
    ckpt_model_name = "./ckpt/" + model_name + '_{epoch:02d}-{val_loss:.4f}'
    csv_name = "./csv/" + model_name
    earlyStopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=15, verbose=0, mode='min')
    mcp_save = ModelCheckpoint(ckpt_model_name + ".h5", save_best_only=True, monitor='val_loss', mode='min')
    reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=7, verbose=1, epsilon=1e-4, mode='min')

    if init:
        bigger_history = model.fit(tr_data, tr_target,
                                   epochs=300,
                                   batch_size=40,
                                   validation_data=(te_data, te_target),
                                   callbacks=[mcp_save],
                                   verbose=2)
    else:
        bigger_history = model.fit(tr_data, tr_target,
                                   epochs=1000,
                                   batch_size=40,
                                   validation_data=(te_data, te_target),
                                   callbacks=[earlyStopping, mcp_save, reduce_lr_loss],
                                   verbose=2)

    hist = pd.DataFrame(bigger_history.history)
    hist['epoch'] = bigger_history.epoch

    hist.to_csv(csv_name + ".csv")
    return model_name, hist


def plot_history_loss(histories, key='categorical_crossentropy'):
    plt.figure(figsize=(16, 10))

    for name, history in histories:
        val = plt.plot(history.epoch, history['val_' + key],
                       '--', label=name.title() + ' Val')
        plt.plot(history.epoch, history[key], color=val[0].get_color(),
                 label=name.title() + ' Train')

    plt.xlabel('Epochs')
    plt.ylabel(key.replace('_', ' ').title())
    plt.legend()

    plt.xlim([0, max(history.epoch)])
    plt.title(name)
    plt.savefig("./plot/ce_" + name + ".png")
    plt.show()


def plot_history_acc(histories, key='categorical_crossentropy'):
    plt.figure(figsize=(16, 10))

    for name, history in histories:
        val = plt.plot(history.epoch, history['val_acc'],
                       '--', label=name.title() + ' Val_acc')
        plt.plot(history.epoch, history['acc'], color=val[0].get_color(),
                 label=name.title() + ' Train_acc')

    plt.xlabel('Epochs')
    plt.ylabel('Acc')
    plt.legend()

    plt.xlim([0, max(history.epoch)])
    plt.title(name)
    plt.savefig("./plot/acc_" + name + ".png")
    plt.show()


def plot_history_mse(histories, key='mean_squared_error'):
    plt.figure(figsize=(16, 10))

    for name, history in histories:
        val = plt.plot(history.epoch, history['val_' + key],
                       '--', label=name.title() + ' Val')
        plt.plot(history.epoch, history[key], color=val[0].get_color(),
                 label=name.title() + ' Train')

    plt.xlabel('Epochs')
    plt.ylabel(key.replace('_', ' ').title())
    plt.legend()

    plt.xlim([0, max(history.epoch)])
    plt.title(name)
    plt.savefig("./plot/ce_" + name + ".png")
    plt.show()


def plot_history_optimizer(histories, key='categorical_crossentropy'):
    plt.figure(figsize=(16, 10))

    for name, history in histories:
        val = plt.plot(history.epoch, history[key], label=name.title() + ' Train')

        plt.plot(history.epoch, history['val_acc'], '--', color=val[0].get_color(), label=name.title() + ' Val_acc')

    plt.xlabel('Epochs')
    plt.ylabel(key.replace('_', ' ').title())
    plt.legend()

    plt.xlim([0, max(history.epoch)])
    plt.title('_'.join(name.split("_")[0:3]) + "_optimizer")
    plt.savefig("./plot/ce_" + '_'.join(name.split("_")[0:3]) + "_optimizer.png")
    plt.show()


if __name__ == "__main__":
    os.makedirs("./ckpt/", exist_ok=True)
    os.makedirs("./csv/", exist_ok=True)
    os.makedirs("./plot/", exist_ok=True)

    data_path = "./hw2_data"
    dataset = ['p1', 'p2']

    # Change hidden layers
    # train_1(data_path, dataset)

    # Change activation function
    # train_2(data_path, dataset)

    # mse with sigmoid test
    #train_2_1(data_path, dataset)

    # optimizer
    #train_3(data_path, dataset)

    # train specific mlp model
    train_best(data_path, dataset)
