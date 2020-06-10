import numpy as np
import tensorflow as tf
import glob
import pandas as pd
import matplotlib.pyplot as plt
import os
import sklearn.mixture as mixture
from sklearn.model_selection import train_test_split

def load_datasets_by_label(tr_data, te_data, label ):#, activity_class="ALL", one_hot=False):
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
    tr_data_by_label = tr_data[np.where(tr_data[:,6]==str(label))]
    te_data_by_label  = te_data[np.where(te_data[:,6]==str(label))]




    return tr_data_by_label[:,:6], te_data_by_label[:,:6]

def load_datasets(data_path, npz=None ):#, activity_class="ALL", one_hot=False):
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
            cnt += 1
            row = np.loadtxt(file)
            file_index, user_id, activity_class = os.path.basename(os.path.splitext(file)[0]).split("_")
            np_lables = np.array([activity_class, user_id,file_index])

            no_lables_reshape = np.repeat([np_lables],row.shape[0], 0)
            np_p1_tr_a = np.column_stack([row, no_lables_reshape])

            rowlist.extend(np_p1_tr_a)
            # np_all_data = np.vstack([np_all_data,np_p1_tr_a])
            # del row
            # del np_p1_tr_a
            # del np_lables
            # del no_lables_reshape
            #print(np_p1_tr_a)
            if cnt % 100 == 0:
                print(cnt)
            #     break
        np_all_data = np.array(rowlist)
        np.savez("./npz/activity_nparray.npz", data = np_all_data)
        print("save complete")
    else:
        np_all_data = np.load("./npz/activity_nparray.npz")['data']
    tr_data, te_data,  = train_test_split(np_all_data,test_size=0.4,shuffle=False,random_state=1004)


    # split orderly


    return tr_data, te_data


def define_GMM(n_mixture, cv_type):
    """Create Guassian mixture model with parameter

    Args:
        n_mixture (string) : # of gaussian component
        cv_type(string) : covariance type (‘full’,‘tied’, ‘diag’, ‘spherical’)

    Returns:
        mixture.GaussianMixture model

    """
    return mixture.GaussianMixture(n_components=n_mixture, covariance_type=cv_type, verbose=2, init_params = 'kmeans')


def gmm_training_plotting(GMM_model, tr_data, te_data , labels, parameter, cv_types):
    """Gmm training and plotting

    Args:
        GMM_model (dict) : empty
        dataset (list) : dataset name (p1, p2)
        labels (list) : target value (0, 1)
        parameter : config class
        cov_type (list)  : covariance type list (‘full’,‘tied’, ‘diag’, ‘spherical’)

    Returns:
        list of dictionary : results of accuracy bu gmm
    """

    n_mixture_config = dict()
    tr_te_data_by_label = dict()
    for n_mixture in parameter.n_mixture:
        # gmm training
        gmm_models = dict()
        name = "alpha"
        for cov_type in cv_types:

            for label in labels:
                #for label in labels:
                model_id = name + "_" + str(label) + "_" + str(cov_type)
                # define gmm model
                GMM_model[model_id] = define_GMM(n_mixture, cov_type)
                # load data
                tr_data_by_label, te_data_by_label = load_datasets_by_label(tr_data, te_data, label)
                tr_te_data_by_label[label] = (tr_data_by_label, te_data_by_label)

                _tr_data_by_label = tr_te_data_by_label[label][0] # 0 is train 1 is test
                #_te_data_by_label = tr_te_data_by_label[label][1]
                #sub_train_data = tr_data[np.where(tr_data[:, 2] == label)]
                # gmm training
                GMM_model[model_id].fit(_tr_data_by_label)
                # save gmm model to dictionary
                gmm_models[model_id] = GMM_model[model_id]
            n_mixture_config[str(n_mixture)] = gmm_models

    line_plot_list = list()
    for n_mixture in parameter.n_mixture:
        GMM_model = n_mixture_config[str(n_mixture)]
        for cov_type in cv_types:
            # gmm plotting
            line_plot_dict = dict()
            #for name in dataset:
            for label in labels:
                #tr_data, te_data = load_datasets(data_path, label)
                # preprocessing data
                sub_train_data = tr_data[np.where(tr_data[:, 2] == label)]
                model_id = name + "_" + str(label) + "_" + str(cov_type)

                tr_data_by_label = tr_te_data_by_label[label][0]
                te_data_by_label = tr_te_data_by_label[label][1]

                n_mixture, name_type, te_acc_percent = validate_gmm(GMM_model, te_data_by_label, name, "test", n_mixture, cov_type, label)
                n_mixture, name_type, tr_acc_percent = validate_gmm(GMM_model, tr_data_by_label, name, "train", n_mixture, cov_type, label)
                #line_plot_list.append([n_mixture, te_acc_percent, tr_acc_percent ])
                line_plot_dict['n_component'] = n_mixture
                line_plot_dict['cov_type'] = cov_type
                line_plot_dict[name + "test"] = te_acc_percent
                line_plot_dict[name + "train"] = tr_acc_percent
                line_plot_list.append(line_plot_dict)
    print("Done")
    return line_plot_list


def validate_gmm(GMM_model, data, name, type, n_mixture, cov_type, label):
    """Validation using test dataset

    Args:
        GMM_model (dict) : # of gmm model dictionary
        data (ndarray) :
        name (string) : dataset name (p1, p2)
        type : data type (test)
        n_mixture (int) : # gaussian components
        cov_type (string)  : covariance type (‘full’,‘tied’, ‘diag’, ‘spherical’)

    Returns:
        int : n_mixture
        string : dataset name
        float : accuracy
    """
    ACC = 0
    test_data = data
    test_target = int(label)-1 # label 1 is 0

    # for i, data in enumerate(test_data):
    #     data = np.expand_dims(data, axis=0)
    candidates = []
    model_keys = {k: v for k, v in GMM_model.items() if k.startswith(name) and k.endswith(cov_type)}
    for model_name in model_keys:
        scores = GMM_model[model_name].score_samples(data)
        # print("label {} : {}".format(label, score))
        candidates.append(scores)
    candidates = np.array(candidates)
    estimated_target = np.argmax(candidates, axis=0)

    # if i % 1000 == 0:
    #     print("Estimated: {}, True: {} / Total Count: ({}/{})\n".format(estimated_target, test_target, i, len(test_data)), end=' ' * 5)

    # if test_target == estimated_target:
    #     # print("correct!")
    #     ACC += 1
    ACC = np.count_nonzero(estimated_target == test_target)

    acc_percent = (ACC / len(test_data) * 100.)
    print("{0} n_component {1} :  {2} {3} {5} Dataset -> ACC:{4:.2f}".format(n_mixture, cov_type, name, type, acc_percent, label))
    return n_mixture, name+type, acc_percent

def gmm_train(tr_data, te_data , cv_types = ['spherical', 'tied', 'diag', 'full']):
    """Gmm train methond

    Args:
        dataset (list) : dataset name (p1, p2,)
        cv_type (list) : covariance types ('spherical', 'tied', 'diag', 'full')

    Returns:
        None
    """

    GMM_model = {}

    labels = dataset
    parameter = Config()

    line_graph_dataset = gmm_training_plotting(GMM_model, tr_data, te_data , labels, parameter, cv_types)

    line_df = pd.DataFrame(line_graph_dataset)
    #line_df["nComponent_cov"] = line_df["n_component"].astype(str) +"-" + line_df["cov_type"].astype(str).str.slice(0,4)

    #line_df.set_index('nComponent_cov', inplace=True)
    #cov_list = line_df["cov_type"].unique()
    #line_df.drop(["n_component", "cov_type"], axis = 1, inplace=True)

    # save csv
    line_df.to_csv("./csv/gmm_train"+str(parameter.n_mixture[-1]) + "_result.csv")
    #plot_bar_graph(line_df, cov_list)
    #plot_line_graph(line_df, cov_list)
    print("done")

class Config():
    """
    Config class for gmm
    """
    def __init__(self):
        """Init for config class

        Args:
            None

        Returns:
            None
        """
        # # of minture
        #self.n_mixture = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
        self.n_mixture = [11,12,13,14,20,25,29]
        # flag for plotting
        self.show_all_plot = True
        # flag for save csv file
        self.save_plot = True

def exam2(tr_data, te_data ):
    """exam2 train gmm with a single variance and multiple muxtures

    Args:
        dataset (list) : dataset name (p1, p2,all)

    Returns:
        None
    """
    gmm_train(tr_data, te_data ,cv_types=['spherical', 'tied', 'diag', 'full'])
    #gmm_train(tr_data, te_data, cv_types=['full'])


if __name__ == "__main__":
    os.makedirs("./csv/", exist_ok=True)
    os.makedirs("./plot/", exist_ok=True)
    os.makedirs("./npz/", exist_ok=True)
    dataset = ['1', '2', '3', '4', '5', '6']
    #dataset = ['1', '2']

    data_path = "./mlpr20_project_train_data"

    tr_data, te_data = load_datasets(data_path, True)

    exam2(tr_data, te_data )
    #for i in range(6):
    load_datasets(data_path, True)