import autokeras as ak
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import tensorflow as tf
#import matplotlib.pyplot as plt
import os
import pandas as pd


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


if __name__ == "__main__":
    os.makedirs("./ckpt/", exist_ok=True)
    os.makedirs("./csv/", exist_ok=True)
    os.makedirs("./plot/", exist_ok=True)
    os.makedirs("./auto_ckpt/", exist_ok=True)

    checkpoint_path = "./auto_ckpt/cp2.hdf5"

    from autokeras.keras_layers import CategoricalEncoding as layer_module
    from autokeras.keras_layers import Sigmoid as sg

#    embedding_layer = layer_module.CategoricalEncoding()
    cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path,
                                                     save_best_only=True,
                                                     verbose=1)
    data_path = "./hw2_data"
    dataset = ['p1', 'p2']
    # Initialize the classifier.
    tr_data, tr_target, te_data, te_target = load_datasets(data_path, 'p2')
    clf = ak.StructuredDataClassifier(max_trials=1000)
    # x is the path to the csv file. y is the column name of the column to predict.
    clf.fit(x=tr_data, y=tr_target, validation_data=(te_data, te_target), callbacks=[cp_callback])
    #test_set = tf.data.Dataset.from_tensor_slices(((te_data,), (te_target,)))
    print(clf.evaluate(te_data, te_target))

    model_get = clf.export_model()
    print(model_get)
    model_get.save("autokeras_best_model.h5")


    #new_auto_keras_model = tf.keras.models.load_model("autokeras_best_model.h5", custom_objects ={"CategoricalEncoding" :embedding_layer } )
    new_auto_keras_model = tf.keras.models.load_model("autokeras_best_model.h5",
                                                     custom_objects={"CategoricalEncoding": layer_module,
                                                                     "Sigmoid" : sg})
    print(model_get.summary())
    print(new_auto_keras_model.summary())



