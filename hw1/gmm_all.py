
import sklearn.mixture as mixture
import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg
import itertools
import matplotlib as mpl
import pandas as pd
import os

def plot_distribution(model, dataset, label, name):
    parameter = Config()
    x = np.linspace(-1.5, 1.5)
    y = np.linspace(-1.5, 1.5)  # 간격을 나눠주고
    X, Y = np.meshgrid(x, y)  # mesh로 그림을 그리고
    XX = np.array([X.ravel(), Y.ravel()]).T
    Z = model.score_samples(XX)
    Z = Z.reshape(X.shape)
    fig, ax = plt.subplots(1,1,figsize=(5,5))

    # contour 그래프
    CS = ax.contour(X, Y,
                    Z,
                    levels=np.linspace(Z.reshape(-1, 1).min(), Z.reshape(-1, 1).max(), 20),
                    antialiased=False
                    )

    CB = fig.colorbar(CS)

    colors = ['steelblue', 'orange']
    markers = ['o','x']
    #labels = [0,1]

    #for i , label in enumerate(labels):
    X = dataset[:,0]
    Y = dataset[:,1]
    ax.set_xlim((-1.5, 1.5))
    ax.set_ylim((-1.5, 1.5))
    ax.scatter(X, Y, marker=markers[label], color=colors[label], label=label)
    ax.legend(loc='upper left')

    plt.title('GMM ' + name)

    plt.savefig('./plot/'+ 'GMM_' + name + "_contour.png")
    plt.show()
    plt.close()

color_iter = itertools.cycle(['navy', 'c', 'cornflowerblue', 'gold',
                              'darkorange'])
def plot_results(X, Y_, means, covariances,covariance_type, label, title):
    """Draw Ellipse graph and save graph

    Args:
        X (ndarray) : X, Y data
        Y_ (ndarray) : predict values
        means (ndarray): mean array,
        covariances (ndarray) : covariance array
        covariance_type (string) : covariance type
        label (int) : target value
        title (string) : title

    Returns:
        None
         . just draw plot graph and save

    """

    colors = ['steelblue', 'orange']
    markers = ['o','x']
    fig, ax = plt.subplots(1,1,figsize=(5,5))

    # Make multi demension arraies if covarian type is tied
    if covariance_type == 'tied':
        cov_list = list()
        for i in range(means.shape[0]):
            cov_list.append(covariances)
        covariances = np.array(cov_list)

    for i, (mean, covar, color) in enumerate(zip(
            means, covariances, color_iter)):
        if covariance_type == 'full':
            covariances = covar  # covar[:2, :2]
        elif covariance_type == 'tied':
            covariances = covar  # covar[:2, :2]
        elif covariance_type == 'diag':
            covariances = np.diag(covar[:2])
        elif covariance_type == 'spherical':
            # covariances = np.eye(gmm.means_.shape[1]) * gmm.covariances_[n]
            covariances = np.eye(mean.shape[0]) * covar
        v, w = linalg.eigh(covariances)
        v = 2. * np.sqrt(2.) * np.sqrt(v)
        u = w[0] / linalg.norm(w[0])

        # as the DP will not use every component it has access to
        # unless it needs it, we shouldn't plot the redundant
        # components.
        # if not np.any(Y_ == i):
        #     continue
        # ax.scatter(X[Y_ == i, 0], X[Y_ == i, 1], marker=markers[label], color=colors[label], label=label)

        # plot all points
        ax.scatter(X[:,0], X[:,1], marker=markers[label], color=colors[label], label=label)

        # Plot an ellipse to show the Gaussian component
        angle = np.arctan(u[1] / u[0])
        angle = 180. * angle / np.pi  # convert to degrees
        ell = mpl.patches.Ellipse(mean, v[0], v[1], 180. + angle, color=color)
        ell.set_clip_box(ax.bbox)
        ell.set_alpha(0.5)
        ax.add_artist(ell)

    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    plt.title(title)
    plt.savefig('./plot/'+ 'GMM_' + title + "_ellip.png")
    plt.show()
    plt.close()

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

    return np_p_tr_a, np_p_te_a

def define_GMM(n_mixture, cv_type):
    """Create Guassian mixture model with parameter

    Args:
        n_mixture (string) : # of gaussian component
        cv_type(string) : covariance type (‘full’,‘tied’, ‘diag’, ‘spherical’)

    Returns:
        mixture.GaussianMixture model

    """
    return mixture.GaussianMixture(n_components=n_mixture, covariance_type=cv_type, verbose=2)

def define_GMM_init(n_mixture, cv_type, init_params='random', warm_start=False, n_init=1):
    """Create Guassian mixture model with parameter and changing initial parameters

    Args:
        n_mixture (string) : # of gaussian component
        cv_type (string) : covariance type (‘full’,‘tied’, ‘diag’, ‘spherical’)
        init_params (string) : Weight initial parameter ('random','kmeans' : default random)
        warm_start (bool) : Using previous weight (default False)
        n_init(int) : #  initializations to perform

    Returns:
        mixture.GaussianMixture model

    """
    return mixture.GaussianMixture(n_components=4, tol = 1e-7, covariance_type=cv_type,
                                   verbose=4, n_init= n_init, init_params=init_params, max_iter=500, warm_start=warm_start)

def dataset_plot(title, data):
    """Plotting scatter plot for training data

    Args:
        title(str) : plot title
        data(Dict) : Dictionary {'p1' : (x,y,t), 'p2' : (x,y,t)}

    Returns:
        None

    """
    fig, ax = plt.subplots(1,1,figsize=(5,5))
    colors = ['steelblue', 'orange']
    markers = ['o','x']
    labels = [0,1]

    for i , label in enumerate(labels):
        X = data[np.where(data[:,2]==label)][:,0]
        Y = data[np.where(data[:,2]==label)][:,1]
        ax.set_xlim((-1.5, 1.5))
        ax.set_ylim((-1.5, 1.5))
        ax.scatter(X, Y, marker=markers[i], color=colors[i], label=label)
    ax.legend(loc='upper left')

    plt.title(title)
    plt.show()
    plt.pause(0.05)
    plt.close()

def validate_gmm(GMM_model, data, name, type, n_mixture, cov_type):
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
    test_data = data[:, 0:2]
    test_target = data[:, 2]

    for i, data in enumerate(test_data):
        data = np.expand_dims(data, axis=0)
        candidates = []
        model_keys = {k: v for k, v in GMM_model.items() if k.startswith(name) and k.endswith(cov_type)}
        for model_name in model_keys:
            score = GMM_model[model_name].score(data)
            # print("label {} : {}".format(label, score))
            candidates.append(score)
        candidates = np.array(candidates)
        estimated_target = np.argmax(candidates)
        # print("Estimated: {}, True: {}\n".format(estimated_target, test_target[i]), end=' ' * 5)

        if test_target[i] == estimated_target:
            # print("correct!")
            ACC += 1

    acc_percent = (ACC / len(test_target) * 100.)
    print("{0} n_component {1} :  {2} {3} Dataset -> ACC:{4:.2f}".format(n_mixture, cov_type, name, type, acc_percent))
    return n_mixture, name+type, acc_percent

def gmm_training_plotting(GMM_model, dataset, labels, parameter, cv_types):
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
    for n_mixture in parameter.n_mixture:
        # gmm training
        gmm_models = dict()
        for cov_type in cv_types:

            for name in dataset:
                for label in labels:
                    model_id = name + "_" + str(label) + "_" + str(cov_type)
                    # define gmm model
                    GMM_model[model_id] = define_GMM(n_mixture, cov_type)
                    # load data
                    tr_data, te_data = load_datasets(data_path, name)
                    sub_train_data = tr_data[np.where(tr_data[:, 2] == label)]
                    # gmm training
                    GMM_model[model_id].fit(sub_train_data[:, 0:2])
                    # save gmm model to dictionary
                    gmm_models[model_id] = GMM_model[model_id]
            n_mixture_config[str(n_mixture)] = gmm_models

    line_plot_list = list()
    for n_mixture in parameter.n_mixture:
        GMM_model = n_mixture_config[str(n_mixture)]
        for cov_type in cv_types:
            # gmm plotting
            line_plot_dict = dict()
            for name in dataset:
                for label in labels:
                    tr_data, te_data = load_datasets(data_path, name)
                    # preprocessing data
                    sub_train_data = tr_data[np.where(tr_data[:, 2] == label)]
                    model_id = name + "_" + str(label) + "_" + str(cov_type)
                    if parameter.show_all_plot:
                        plot_distribution(GMM_model[model_id], sub_train_data, label, str(n_mixture) + " mixture " + model_id)

                        plot_results(sub_train_data[:,0:2], GMM_model[model_id].predict(sub_train_data[:,0:2]),
                                    GMM_model[model_id].means_, GMM_model[model_id].covariances_,
                                    GMM_model[model_id].covariance_type, label, str(n_mixture) + " mixture " + model_id)

                tr_data, te_data = load_datasets(data_path, name)
                n_mixture, name_type, te_acc_percent = validate_gmm(GMM_model, te_data, name, "test", n_mixture, cov_type)
                n_mixture, name_type, tr_acc_percent = validate_gmm(GMM_model, tr_data, name, "train", n_mixture, cov_type)
                #line_plot_list.append([n_mixture, te_acc_percent, tr_acc_percent ])
                line_plot_dict['n_component'] = n_mixture
                line_plot_dict['cov_type'] = cov_type
                line_plot_dict[name + "test"] = te_acc_percent
                line_plot_dict[name + "train"] = tr_acc_percent
            line_plot_list.append(line_plot_dict)
    print("Done")
    return line_plot_list

def init_model_parameters(dataset, labels, init_params):
    """ Check convergence time using weight initalizaion methond random and kmeans

    Args:
        dataset (list) : dataset name (p1, p2, all)
        labels (list) : target value (0, 1)
        init_params (string) : init parameters (random, kmeans)

    Returns:
        None
    """

    print("Start with init param {0}".format(init_params))
    #n_mixture_config = dict()
    best_n_mixture=30
    best_cov = 'full'
    for name in dataset:
        for label in labels:
            model_id = name + "_" + str(label) + "_" + str(best_cov)
            print("gmm model training check convergence -> {0}".format(model_id))
            # define gmm model
            gmm = define_GMM_init(best_n_mixture,best_cov, init_params, )
            # load data
            tr_data, te_data = load_datasets(data_path, name)
            sub_train_data = tr_data[np.where(tr_data[:, 2] == label)]
            # gmm training
            log = gmm.fit(sub_train_data[:, 0:2])

def warm_model_start(dataset, labels, init_params):
    """ Check warm start time

    Args:
        dataset (list) : dataset name (p1, p2, all)
        labels (list) : target value (0, 1)
        init_params (string) : init parameters (random, kmeans)

    Returns:
        None
    """
    best_n_mixture=30
    best_cov = 'full'

    # define gmm model
    print("Start all data fit")
    gmm = define_GMM_init(best_n_mixture, best_cov, init_params, True)
    # load data
    tr_data, te_data = load_datasets(data_path, 'all')
    sub_train_data = tr_data
    # gmm training
    gmm.fit(sub_train_data[:, 0:2])

    for name in dataset[0:2]:
        for label in labels:
            print("Start target {0}  {1} data fit".format(label,name))
            tr_data, te_data = load_datasets(data_path, name)
            sub_train_data = tr_data[np.where(tr_data[:, 2] == label)]
            # gmm training
            gmm.fit(sub_train_data[:, 0:2])

def plot_line_graph(line_df, cov_list):
    """Draw line graph for all gmms accuracy

    Args:
        line_df (DataFrame) : The result of differemt gmm accuracy
        cov_list (list) : predict values

    Returns:
        None
         . just draw plot graph and save

    """
    # Line Graph by matplotlib with wide-form DataFrame
    fig, ax = plt.subplots(1,1,figsize=(10,10))
    ax.plot(line_df.index, line_df.p1test.values, marker='s', color='r', label='p1test' )
    ax.plot(line_df.index, line_df.p1train.values, marker='o', color='#ffa500', label='p1train')
    ax.plot(line_df.index, line_df.p2test.values, marker='*', color='b', label='p2test')
    ax.plot(line_df.index, line_df.p2train.values, marker='+', color='#00bfff', label = 'p2train')

    plt.title('Accuracy', fontsize=20)
    plt.ylabel('Acc', fontsize=14)
    plt.xlabel('n_component', fontsize=14)
    plt.legend(fontsize=12, loc='best')
    xlabels = [str(x).split('-')[0] for x in line_df.index.unique()]
    ax.set_xticklabels(xlabels)
    plt.savefig("./plot/GMM_ALL_"+'-'.join(cov_list)+"_line.png")
    plt.show()
    plt.close()

def plot_bar_graph(line_df, cov_list):
    """Draw bar graph for all gmms accuracy

    Args:
        line_df (DataFrame) : The result of differemt gmm accuracy
        cov_list (list) : predict values

    Returns:
        None
         . just draw plot graph and save

    """
    line_df.plot(kind='bar')
    if 'nComponent_cov' in line_df.columns:
        Xaxis_label = line_df['nComponent_cov'].unique()
        X = list(range(len(line_df.index)))
        plt.xticks(X, Xaxis_label)
    plt.title('Accuracy', fontsize=20)
    plt.savefig("./plot/GMM_ALL_"+'-'.join(cov_list)+"_bar2.png")
    plt.show()

def gmm_train(dataset, cv_types = ['spherical', 'tied', 'diag', 'full']):
    """Gmm train methond

    Args:
        dataset (list) : dataset name (p1, p2,)
        cv_type (list) : covariance types ('spherical', 'tied', 'diag', 'full')

    Returns:
        None
    """

    GMM_model = {}

    labels = [0,1]
    parameter = Config()

    line_graph_dataset = gmm_training_plotting(GMM_model, dataset, labels, parameter, cv_types)

    line_df = pd.DataFrame(line_graph_dataset)
    line_df["nComponent_cov"] = line_df["n_component"].astype(str) +"-" + line_df["cov_type"].astype(str).str.slice(0,4)

    line_df.set_index('nComponent_cov', inplace=True)
    cov_list = line_df["cov_type"].unique()
    line_df.drop(["n_component", "cov_type"], axis = 1, inplace=True)

    # save csv
    line_df.to_csv("./csv/gmm_train"+str(parameter.n_mixture[-1])+"-".join(cov_list) +"_result.csv")
    plot_bar_graph(line_df, cov_list)
    plot_line_graph(line_df, cov_list)
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
        self.n_mixture = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
        # flag for plotting
        self.show_all_plot = True
        # flag for save csv file
        self.save_plot = True

def exam1(dataset):
    """exam1 plot dataset

    Args:
        dataset (list) : dataset name (p1, p2,all)

    Returns:
        None
    """
    for name in dataset:
        tr_data, te_data = load_datasets(data_path, name)
        dataset_plot(name + " train data", tr_data)

def exam2(dataset):
    """exam2 train gmm with a single variance and multiple muxtures

    Args:
        dataset (list) : dataset name (p1, p2,all)

    Returns:
        None
    """
    gmm_train(dataset,cv_types=['spherical'])

def exam3(dataset):
    """exam3 train gmm with multiple variance types and multiple muxtures

    Args:
        dataset (list) : dataset name (p1, p2,all)

    Returns:
        None
    """
    gmm_train(dataset)

def exam4(dataset):
    """ evaluate convergence time using initial parameters

    Args:
        dataset (list) : dataset name (p1, p2,all)

    Returns:
        None
    """
    GMM_model = {}
    labels = [0,1]
    init_params = 'random'
    line_graph_dataset = init_model_parameters(dataset, labels, init_params )
    init_params = 'kmeans'
    line_graph_dataset = init_model_parameters(dataset, labels, init_params )

def exam5(dataset):
    """ evaluate convergence time using warm start parameter

    Args:
        dataset (list) : dataset name (p1, p2,all)

    Returns:
        None
    """
    #GMM_model = {}
    labels = [0,1]
    init_params = 'random'
    line_graph_dataset = warm_model_start(dataset, labels, init_params)

def redraw_lineplot():
    """ redraw plot using result file(csv)

    Args:
        None

    Returns:
        None
    """
    line_df = pd.read_csv("csv_3/gmm_train20spherical-tied-diag-full_result.csv")
    cv_types = ['spherical', 'tied', 'diag', 'full']
    line_df = line_df.iloc[[32, 33, 34, 35, 36, 37, 38, 39, 56, 57, 58, 59, 60, 61, 62, 63]]
    line_df.drop(["alltest", "alltrain"], axis = 1, inplace=True)
    plot_bar_graph(line_df, cv_types)
    plot_line_graph(line_df, cv_types)

if __name__ == "__main__":
    # plot img directory check
    os.makedirs("./plot/", exist_ok=True)
    # result csv directory check
    os.makedirs("./csv/", exist_ok=True)

    data_path = "./hw1_data/"
    #np_p1_tr_i, np_p1_tr_t, np_p1_te_i, np_p1_te_t = load_datasets(data_path)

    dataset = ['p1', 'p2', 'all']

    # Plotting train dataset
    exam1(dataset)
    dataset.remove('all')
    exam2(dataset)
    exam3(dataset)

    exam4(dataset)
    dataset = ['p1']
    exam5(dataset)
    dataset = ['p2']
    exam5(dataset)

    # redraw_lineplot()





