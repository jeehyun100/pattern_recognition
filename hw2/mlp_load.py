import numpy as np
import tensorflow as tf
import glob
import pandas as pd
import matplotlib.pyplot as plt
import os

from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import PReLU
from tensorflow.keras.layers import ELU
from tensorflow.keras.initializers import TruncatedNormal
from tensorflow.keras.initializers import glorot_normal
from tensorflow.keras.initializers import glorot_uniform
from matplotlib.colors import ListedColormap
from sklearn import neighbors, datasets


def load_datasets(data_path , dataset_name, one_hot=False):
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
    np_p1_tr_i = np.loadtxt(data_path + "/p1_train_input.txt")
    np_p1_tr_t = np.loadtxt(data_path + "/p1_train_target.txt")
    np_p1_te_i = np.loadtxt(data_path + "/p1_test_input.txt")
    np_p1_te_t = np.loadtxt(data_path + "/p1_test_target.txt")

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

    tr_target = np_p_tr_a[:,2]
    te_target = np_p_te_a[:,2]

    if one_hot:
        tr_target = tr_target.reshape(-1, 1)
        te_target = te_target.reshape(-1, 1)
        enc_tr = OneHotEncoder()
        enc_tr.fit(tr_target)
        tr_target = enc_tr.transform(tr_target).toarray()

        enc_te = OneHotEncoder()
        enc_te.fit(te_target)
        te_target = enc_tr.transform(te_target).toarray()

    return np_p_tr_a[:,0:2],tr_target, np_p_te_a[:,0:2],te_target


def load_model():
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

    new_model = tf.keras.models.load_model('.mdl_wts.hdf5')
    loss, acc,_ = new_model.evaluate(te_data,  te_target_onehot, verbose=2)
    print("복원된 모델의 정확도: {:5.2f}%".format(100*acc))

    example_batch = te_data[:1]
    example_target = te_target[:1,0]
    example_result = new_model.predict(example_batch)

    print("predict {0} target{1}".format(np.argmax(example_result),example_target))


def test1(model_path, data_path):
    file_list = glob.glob(model_path + "*")
    split_file_list = [(test.replace("./ckpt/", "").split("_")[0] + "_" +
                       test.replace("./ckpt/", "").split("_")[1] + "_" +
                       test.replace("./ckpt/", "").split("_")[2]
                       , test.replace("./ckpt/", "").split("-")[0].split("_")[-1]
                       , test.replace("./ckpt/", "").split("-")[1].replace(".h5", "")) for test in file_list]
    df_file_list = pd.DataFrame(split_file_list)

    mlp_values = df_file_list[0].unique()
    test_value_max = list()
    for value in mlp_values:
        get_first_row = df_file_list.loc[(df_file_list[0] == value)].sort_values([2], ascending=[True]).iloc[0]
        test_value_max.append(get_first_row)
        model_path = "./ckpt/" + get_first_row[0]+"_"+ get_first_row[1] +"-" + get_first_row[2] + ".h5"
        mlp_exam_load(data_path,get_first_row[0].split("_")[-1], model_path, get_first_row[0] )

    print(test_value_max)


def test2(model_path, data_path):
    file_list = glob.glob(model_path + "*")
    split_file_list = [(test.replace("./ckpt/", "").split("_")[0]+"_" +
                       test.replace("./ckpt/", "").split("_")[1]+"_" +
                       test.replace("./ckpt/", "").split("_")[2]+"_" +
                        test.replace("./ckpt/", "").split("_")[3]
                       , test.replace("./ckpt/", "").split("-")[0].split("_")[-1]
                       , test.replace("./ckpt/", "").split("-")[1].replace(".h5", "")) for test in file_list]
    df_file_list = pd.DataFrame(split_file_list)

    mlp_values = df_file_list[0].unique()
    test_value_max = list()
    for value in mlp_values:
        get_first_row = df_file_list.loc[(df_file_list[0] == value)].sort_values([2], ascending=[True]).iloc[0]
        test_value_max.append(get_first_row)
        model_path = "./ckpt/" + get_first_row[0]+"_"+ get_first_row[1] +"-" + get_first_row[2] + ".h5"
        mlp_exam_load(data_path,get_first_row[0].split("_")[2], model_path, get_first_row[0] )

    print(test_value_max)


def test_best(model_path, data_path):
    file_list = glob.glob(model_path + "*")
    for mp in file_list:
        if 'p1' in mp:
            name = 'p1'
        if 'p2' in mp:
            name = 'p2'
        mlp_best_load(data_path,name, mp, mp.replace("./ckpt/",""))


def load_autokeras_model():

    import autokeras as ak
    model_path = "./auto_ckpt/cp.hdf5"
    new_model = tf.keras.models.load_model(model_path, custom_objects={'CategoricalEncoding': ak.CategoricalToNumerical()})
    print(new_model)

def mlp_exam_load(data_path, name, model_path, title):

    tr_data, tr_target, te_data, te_target = load_datasets(data_path, name, True)
    # Create color maps
    cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
    cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

    h = .01 # step size in the mesh
    x_min, x_max = te_data[:, 0].min() - 0.2, te_data[:, 0].max() + 0.2
    y_min, y_max = te_data[:, 1].min() - 0.2, te_data[:, 1].max() + 0.2
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    np_test = np.c_[xx.ravel(), yy.ravel()]

    if 'mlp_9' in model_path:
        print("model_path")
    if "LeakyReLU" in title.split("_")[3]:
        new_model = tf.keras.models.load_model(model_path, custom_objects={'LeakyReLU': LeakyReLU})
    elif "ELU" in title.split("_")[3]:
        new_model = tf.keras.models.load_model(model_path, custom_objects={'ELU': ELU})
    elif "PReLU" in title.split("_")[3]:
        new_model = tf.keras.models.load_model(model_path, custom_objects={'PReLU': PReLU})
    else:
        new_model = tf.keras.models.load_model(model_path)

    Z = new_model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z1 = np.argmax(Z, axis=1)
    Z1 = Z1.reshape(xx.shape)
    plt.figure()
    plt.pcolormesh(xx, yy, Z1, cmap=cmap_light)

    loss, acc,_ = new_model.evaluate(te_data,  te_target, verbose=2)
    # Plot also the training points
    plot_te_target = np.argmax(te_target, axis=1)
    plt.scatter(te_data[:, 0], te_data[:, 1], c=plot_te_target, cmap=cmap_bold)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title(title)
    plt.savefig("./plot/boundary_" + title + ".png")
    plt.show()


def mlp_best_load(data_path, name, model_path, title):

    tr_data, tr_target, te_data, te_target = load_datasets(data_path, name, True)
    # Create color maps
    cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
    cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

    h = .01 # step size in the mesh
    x_min, x_max = te_data[:, 0].min() - 0.2, te_data[:, 0].max() + 0.2
    y_min, y_max = te_data[:, 1].min() - 0.2, te_data[:, 1].max() + 0.2
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    np_test = np.c_[xx.ravel(), yy.ravel()]
    new_model = tf.keras.models.load_model(model_path, custom_objects={'PReLU': PReLU})

    Z = new_model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z1 = np.argmax(Z,axis=1)
    # Put the result into a color plot
    Z1 = Z1.reshape(xx.shape)
    plt.figure()
    plt.pcolormesh(xx, yy, Z1, cmap=cmap_light)
    loss, acc,_ = new_model.evaluate(te_data,  te_target, verbose=2)
    print("{0} 복원된 모델의 정확도: {1:5.2f}%".format(name, 100*acc))

    plot_te_target = np.argmax(te_target, axis=1)
    plt.scatter(te_data[:, 0], te_data[:, 1], c=plot_te_target, cmap=cmap_bold)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title(title)
    plt.savefig("./plot/boundary_" + title + ".png")
    plt.show()

if __name__ == "__main__":
    os.makedirs("./plot/", exist_ok=True)

    data_path = "./hw2_data"
    dataset = ['p1', 'p2']

    model_path = "./ckpt_p1_92/mlp_1_p1_relu.h5"
    mlp_best_load(data_path, 'p1', model_path, "mlp_p1_prelu")

    model_path = "./ckpt_p2_94.54/mlp_1_p2_relu.h5"
    mlp_best_load(data_path, 'p2', model_path, "mlp_p1_prelu")

    # pip install tensorflow==2.1.0 and autokeras if you want to use auto keras
    #load_autokeras_model()
