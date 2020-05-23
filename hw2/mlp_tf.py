import numpy as np
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import ReduceLROnPlateau
import pandas as pd
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Activation
from tensorflow.keras.layers import BatchNormalization



class SimpleMLP(Model):

    def __init__(self, use_bn=False, use_dp=False, num_classes=2):
        super(SimpleMLP, self).__init__(name='mlp')
        self.use_bn = use_bn
        self.use_dp = use_dp
        self.num_classes = num_classes

        self.dense1 = Dense(32, activation='relu')
        self.dense2 = Dense(num_classes, activation='softmax')
        if self.use_dp:
            self.dp = Dropout(0.5)
        if self.use_bn:
            self.bn = BatchNormalization(axix=-1)
    def call(self, input):
        x = self.dense1(input)
        if self.use_dp:
            x = self.dp(x)
        if self.use_bn:
            x = self.bn(x)
        return self.dense2(x)



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


data_path = "./hw2_data"
name = 'p1'
tr_data, tr_target, te_data, te_target = load_datasets(data_path, name)
tr_target = tr_target.reshape(-1,1)
te_target = te_target.reshape(-1,1)

enc_tr = OneHotEncoder()
enc_tr.fit(tr_target)
tr_target_onehot = enc_tr.transform(tr_target).toarray()

enc_te = OneHotEncoder()
enc_te.fit(te_target)
te_target_onehot = enc_tr.transform(te_target).toarray()

Y_class_recovery = np.argmax(tr_target_onehot, axis=1).reshape(-1,1)


#train_dataset = tf.data.Dataset.from_tensor_slices((tr_data, tr_target_onehot))
#test_dataset = tf.data.Dataset.from_tensor_slices((test_examples, test_labels))

BATCH_SIZE = 64
SHUFFLE_BUFFER_SIZE = 100

#train_dataset = train_dataset.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
#test_dataset = test_dataset.batch(BATCH_SIZE)



def custom_mlp(input_shape, depth, num_classes=2):
    """ResNet Version 2 Model builder [b]

    # Arguments
        input_shape (tensor): shape of input image tensor
        depth (int): number of core convolutional layers
        num_classes (int): number of classes (CIFAR10 has 10)
    # Returns
        model (Model): Keras model instance
    """


    inputs = tf.keras.Input(shape=input_shape)
    # v2 performs Conv2D with BN-ReLU on input before splitting into 2 paths
    #x = Dense(inputs, activation='relu')
    x = Dense(64, activation='relu')(inputs)
    for _ in range(depth):
        x = Dense(64, activation='relu')(x)


    outputs = Dense(num_classes,
                    activation='softmax',
                    kernel_initializer='he_normal')(x)

    # Instantiate model.
    model = Model(inputs=inputs, outputs=outputs)
    return model

def build_model():
  model = tf.keras.Sequential([
    tf.keras.layers.Dense(16, activation='relu', input_shape=[2]),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(256, activation='softmax')
  ])

  optimizer = tf.keras.optimizers.RMSprop(0.001)
  model.compile(optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy','categorical_crossentropy'])
  # model.compile(loss='mse',
  #               optimizer=optimizer,
  #               metrics=['mae', 'mse'])
  return model


#model = build_model()


original_dim = 2
intermediate_dim = 64
latent_dim = 32
units = 32
timesteps = 10
input_dim = 1
batch_size = 16

inputs = tf.keras.Input(batch_shape=(batch_size, timesteps, input_dim))
# #x = layers.Conv1D(32, 3)(inputs)
# hidden = SimpleMLP(name = "1")(inputs)
# hidden2 = SimpleMLP(name = "2")(hidden)
#
# model = Model(inputs, hidden2)
#
#
# # Define encoder model.
# inputs = tf.keras.Input(shape=(original_dim,), name='encoder_input')
#
# model1 = SimpleMLP(inputs, enc_out)
# model= SimpleMLP(enc_out)

#m = SimpleMLP()
input_shape = 2
depth = 10
model = custom_mlp(input_shape=input_shape, depth=depth)

optimizer = tf.keras.optimizers.RMSprop(0.001)
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy', 'categorical_crossentropy'])


print(model.summary())
# checkpoint_path = "training_1/cp.ckpt"
# checkpoint_dir = os.path.dirname(checkpoint_path)




earlyStopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='min')
mcp_save = ModelCheckpoint('.mdl_wts.hdf5', save_best_only=True, monitor='val_loss', mode='min')
reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=7, verbose=1, epsilon=1e-4, mode='min')

#model.fit(Xtr_more, Ytr_more, batch_size=batch_size, epochs=50, verbose=0, callbacks=[earlyStopping, mcp_save, reduce_lr_loss], validation_split=0.25)


# history = model.fit(
#   normed_train_data, train_labels,
#   epochs=64, validation_split = 0.2, verbose=0)

#model.fit(train_dataset, epochs=500, shuffle=True)
bigger_history = model.fit(tr_data, tr_target_onehot,
                                  epochs=1000,
                                  batch_size=40,
                                  validation_data=(te_data, te_target_onehot),
                                  callbacks=[earlyStopping, mcp_save, reduce_lr_loss],
                                  verbose=2)


hist = pd.DataFrame(bigger_history.history)
hist['epoch'] = bigger_history.epoch

print(hist.tail())


def plot_history(histories, key='categorical_crossentropy'):
  plt.figure(figsize=(16,10))

  for name, history in histories:
    val = plt.plot(history.epoch, history.history['val_'+key],
                   '--', label=name.title()+' Val')
    plt.plot(history.epoch, history.history[key], color=val[0].get_color(),
             label=name.title()+' Train')

  plt.xlabel('Epochs')
  plt.ylabel(key.replace('_',' ').title())
  plt.legend()

  plt.xlim([0,max(history.epoch)])
  plt.show()


plot_history([('bigger', bigger_history)])


# |-Trial ID: 24a33eb48545013052ed41ac86f2cbd6
# |-Score: 0.5644444227218628
# |-Best step: 0
# > Hyperparameters:
# |-classification_head_1/dropout_rate: 0.0
# |-optimizer: adam
# |-structured_data_block_1/dense_block_1/dropout_rate: 0.0
# |-structured_data_block_1/dense_block_1/num_layers: 1
# |-structured_data_block_1/dense_block_1/units_0: 16
# |-structured_data_block_1/dense_block_1/units_1: 128
# |-structured_data_block_1/dense_block_1/units_2: 256
# |-structured_data_block_1/dense_block_1/use_batchnorm: True
#WARNING:tensorflow:Unresolved object in checkpoint: (root).optimizer.iter
#WARNING:tensorflow:Unresolved object in checkpoint: (root).optimizer.beta_1
#WARNING:tensorflow:Unresolved object in checkpoint: (root).optimizer.beta_2
#WARNING:tensorflow:Unresolved object in checkpoint: (root).optimizer.decay
#WARNING:tensorflow:Unresolved object in checkpoint: (root).optimizer.learning_rate
#WARNING:tensorflow:A checkpoint was restored (e.g. tf.train.Checkpoint.restore or tf.keras.Model.load_weights) but not all checkpointed values were used. See above for specific issues. Use expect_partial() on the load status object, e.g. tf.train.Checkpoint.restore(...).expect_partial(), to silence these warnings, or use assert_consumed() to make the check explicit. See https://www.tensorflow.org/guide/checkpoint#l


