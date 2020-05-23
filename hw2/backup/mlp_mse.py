import numpy as np
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import ReduceLROnPlateau
import pandas as pd


def load_datasets(data_path , dataset_name):
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
    #if 'p1' in data_type:
    np_p1_tr_i = np.loadtxt(data_path + "/p1_train_input.txt")
    np_p1_tr_t = np.loadtxt(data_path + "/p1_train_target.txt")
    np_p1_te_i = np.loadtxt(data_path + "/p1_test_input.txt")
    np_p1_te_t = np.loadtxt(data_path + "/p1_test_target.txt")

    #elif 'p2' in data_type:
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
        np_p_tr_a = np.vstack([np_p1_tr_a,np_p2_tr_a])
        np_p_te_a = np.vstack([np_p1_te_a,np_p2_te_a])

    return np_p_tr_a[:,0:2],np_p_tr_a[:,2], np_p_te_a[:,0:2],np_p_te_a[:,2]


data_path = "../hw2_data"
name = 'p1'
tr_data, tr_target, te_data, te_target = load_datasets(data_path, name)
# tr_target = tr_target.reshape(-1,1)
# te_target = te_target.reshape(-1,1)
#
# enc_tr = OneHotEncoder()
# enc_tr.fit(tr_target)
# tr_target_onehot = enc_tr.transform(tr_target).toarray()
#
# enc_te = OneHotEncoder()
# enc_te.fit(te_target)
# te_target_onehot = enc_tr.transform(te_target).toarray()
#
# Y_class_recovery = np.argmax(tr_target_onehot, axis=1).reshape(-1,1)


#train_dataset = tf.data.Dataset.from_tensor_slices((tr_data, tr_target_onehot))
#test_dataset = tf.data.Dataset.from_tensor_slices((test_examples, test_labels))

BATCH_SIZE = 64
SHUFFLE_BUFFER_SIZE = 100

#train_dataset = train_dataset.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
#test_dataset = test_dataset.batch(BATCH_SIZE)


def build_model():
  model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=[2]),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1)
  ])

  optimizer = tf.keras.optimizers.RMSprop(0.001)
  model.compile(optimizer=optimizer,
                loss='mse',
                metrics=['mae', 'mse'])
  # model.compile(loss='mse',
  #               optimizer=optimizer,
  #               metrics=['mae', 'mse'])
  return model


model = build_model()


# checkpoint_path = "training_1/cp.ckpt"
# checkpoint_dir = os.path.dirname(checkpoint_path)




earlyStopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='min')
mcp_save = ModelCheckpoint('.mdl_wts_mse.hdf5', save_best_only=True, monitor='val_loss', mode='min')
reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=7, verbose=1, epsilon=1e-4, mode='min')

#model.fit(Xtr_more, Ytr_more, batch_size=batch_size, epochs=50, verbose=0, callbacks=[earlyStopping, mcp_save, reduce_lr_loss], validation_split=0.25)


# history = model.fit(
#   normed_train_data, train_labels,
#   epochs=64, validation_split = 0.2, verbose=0)

#model.fit(train_dataset, epochs=500, shuffle=True)
bigger_history = model.fit(tr_data, tr_target,
                                  epochs=1000,
                                  batch_size=40,
                                  validation_data=(te_data, te_target),
                                  callbacks=[earlyStopping, mcp_save, reduce_lr_loss],
                                  verbose=2)


hist = pd.DataFrame(bigger_history.history)
hist['epoch'] = bigger_history.epoch

print(hist.tail())




def plot_history2(history):
  hist = pd.DataFrame(history.history)
  hist['epoch'] = history.epoch

  plt.figure(figsize=(8,12))

  plt.subplot(2,1,1)
  plt.xlabel('Epoch')
  plt.ylabel('Mean Abs Error [MPG]')
  plt.plot(hist['epoch'], hist['mean_absolute_error'],
           label='Train Error')
  plt.plot(hist['epoch'], hist['val_mean_absolute_error'],
           label = 'Val Error')
  plt.ylim([0,5])
  plt.legend()

  plt.subplot(2,1,2)
  plt.xlabel('Epoch')
  plt.ylabel('Mean Square Error [$MPG^2$]')
  plt.plot(hist['epoch'], hist['mean_squared_error'],
           label='Train Error')
  plt.plot(hist['epoch'], hist['val_mean_squared_error'],
           label = 'Val Error')
  plt.ylim([0,20])
  plt.legend()
  plt.show()

plot_history2(bigger_history)