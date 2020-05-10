
import sklearn.mixture as mixture
import numpy as np
import librosa
import pdb
import glob
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from mpl_toolkits.mplot3d import Axes3D
import matplotlib as cm
from scipy import linalg
import itertools
import matplotlib as mpl
import pandas as pd


def generate_examples(n_mixture, n_samples=300):
    toy_data = []
    for i in range(n_mixture):
        var = np.random.uniform(0,10,2)
        mu = np.random.uniform(-40,40,2)
        dist = np.random.normal(mu, var, (n_samples,2))
        toy_data.append(dist)
        print("var {0}".format(var))
    toy_data = np.vstack(toy_data)
    return toy_data


def define_GMM(n_mixture):
    #sk learn에서 정의되어 있는 mixture
    return mixture.GaussianMixture(n_components=n_mixture, covariance_type='diag') #full


def plot_distribution2(model, X_train):
    x1 = X_train[:,0]
    x2 = X_train[:,1]
    #X, Y = np.meshgrid(x1, x2)  # mesh로 그림을 그리고
    #XX = np.array([X.ravel(), Y.ravel()]).T
    Z = model.score_samples(X_train)
    #Z = Z.reshape(X_train.shape)
    CS = plt.contour(x1, x2, -np.log10(-Z), antialiased=False)  # contour 그래프
    CB = plt.colorbar(CS)
    plt.scatter(X_train[:, 0], X_train[:, 1], .8)
    plt.title('Negative log-likelihood predicted by a GMM')
    plt.axis('tight')
    plt.show()
    plt.close()

def plot_distribution(model, dataset, label, name):
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
    plt.show()
    plt.close()

    # fig = plt.gca(projection='3d')
    # surf = fig.plot_surface(X, Y, -np.log10(-Z), cmap='coolwarm', linewidth=0, antialiased=False)
    # plt.colorbar(surf)
    # plt.title('3D contour of GMM')
    # fig.set_xlabel('x')
    # fig.set_ylabel('y')
    # fig.set_zlabel('z')
    # plt.show()
    # plt.close()

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

    Example:
        >>> r = np.array([1, 1, 1])
        >>> discount_rewards(r, 0.99)
        np.array([1 + 0.99 + 0.99**2, 1 + 0.99, 1])
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

    # if select_type == 'partial':
    #     # Selected by a label value
    #     np_p1_tr_a = np_p1_tr_a[np.where(np_p1_tr_a[:,2] == train_label)]
    # Remove label column
    #return np_p1_tr_a[:,0:2], np_p1_tr_a[:,2]
    return np_p_tr_a, np_p_te_a

def load_test_data(data_path):

    #np_p1_tr_i = np.loadtxt(data_path + "/p1_train_input.txt")
    #np_p1_tr_t = np.loadtxt(data_path + "/p1_train_target.txt")
    np_p1_te_i = np.loadtxt(data_path + "/p1_test_input.txt")
    np_p1_te_t = np.loadtxt(data_path + "/p1_test_target.txt")

    np_p1_te_a = np.column_stack([np_p1_te_i, np_p1_te_t])
    #np_p1_te_a_x = np_p1_te_a[np.where(np_p1_te_a[:,2] == train_label)]
    #np_p1_tr_a_1 = np_p1_tr_a[np.where(np_p1_tr_a[:, 2] == 1)]
    #return np_p1_te_a[:,0:2], np_p1_te_a[:,2]
    return np_p1_te_a

def define_GMM(n_mixture):
    return mixture.GaussianMixture(n_components=n_mixture, covariance_type='full')

def dataset_plot(title, data):
    """Plotting scatter plot for training data

    Args:
        title(str) : plot title
        data(Dict) : Dictionary {'p1' : (x,y,t), 'p2' : (x,y,t)}

    Returns:
        None

    """
    # x = np.linspace(-1.5, 1.5)
    # y = np.linspace(-1.5, 1.5)  # 간격을 나눠주고
    # y = np.linspace(-1.5, 1.5)  # 간격을 나눠주고
    # X, Y = np.meshgrid(x, y)  # mesh로 그림을 그리고
    # #XX = np.array([X.ravel(), Y.ravel()]).T

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
    plt.close()

def validate_gmm(GMM_model, data, name, type, n_mixture):
    ACC = 0
    test_data = data[:, 0:2]
    test_target = data[:, 2]

    for i, data in enumerate(test_data):
        data = np.expand_dims(data, axis=0)
        candidates = []
        model_keys = {k: v for k, v in GMM_model.items() if k.startswith(name)}
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
        # else:
        # print("incorrect...")
    acc_percent = (ACC / len(test_target) * 100.)
    print("{0} n_component :  {1} {2} Dataset -> ACC:{3:.2f}".format(n_mixture, name, type, acc_percent))
    return n_mixture, name+type, acc_percent

def gmm_training_plotting(GMM_model, dataset, labels, parameter):
    n_mixture_config = dict()
    #gmm_models = list()
    for n_mixture in parameter.n_mixture:
        # gmm training
        #gmm_models.clear()
        gmm_models = dict()
        for name in dataset:
            for label in labels:
                GMM_model[name + "_" + str(label)] = define_GMM(n_mixture)
                tr_data, te_data = load_datasets(data_path, name)
                sub_train_data = tr_data[np.where(tr_data[:, 2] == label)]
                GMM_model[name + "_" + str(label)].fit(sub_train_data[:, 0:2])
                gmm_models[name + "_" + str(label)] = GMM_model[name + "_" + str(label)]
        n_mixture_config[str(n_mixture)] = gmm_models

    line_plot_list = list()
    for n_mixture in parameter.n_mixture:
        GMM_model = n_mixture_config[str(n_mixture)]

        # gmm plotting
        line_plot_dict = dict()
        for name in dataset:
            for label in labels:
                tr_data, te_data = load_datasets(data_path, name)
                sub_train_data = tr_data[np.where(tr_data[:, 2] == label)]
                #plot_distribution(GMM_model[name + "_" + str(label)], sub_train_data, label, str(n_mixture) + " mixture " + name + "_" + str(label))

                #plot_results(sub_train_data[:,0:2], GMM_model[name + "_" + str(label)].predict(sub_train_data[:,0:2]),
                #             GMM_model[name + "_" + str(label)].means_, GMM_model[name + "_" + str(label)].covariances_,
                #             label, str(n_mixture) + " mixture " + name + "_" + str(label))

            tr_data, te_data = load_datasets(data_path, name)
            n_mixture, name_type, te_acc_percent = validate_gmm(GMM_model, te_data, name, "test", n_mixture)
            n_mixture, name_type, tr_acc_percent = validate_gmm(GMM_model, tr_data, name, "train", n_mixture)
            #line_plot_list.append([n_mixture, te_acc_percent, tr_acc_percent ])
            line_plot_dict['n_component'] = n_mixture
            line_plot_dict[name + "test"] = te_acc_percent
            line_plot_dict[name + "train"] = tr_acc_percent
        line_plot_list.append(line_plot_dict)
    print("Done")
    return line_plot_list

def make_ellipses(gmm, ax):
    for n in range(gmm.n_components):
        if gmm.covariance_type == 'full':
            covariances = gmm.covariances_[n][:2, :2]
        elif gmm.covariance_type == 'tied':
            covariances = gmm.covariances_[:2, :2]
        elif gmm.covariance_type == 'diag':
            covariances = np.diag(gmm.covariances_[n][:2])
        elif gmm.covariance_type == 'spherical':
            covariances = np.eye(gmm.means_.shape[1]) * gmm.covariances_[n]
        v, w = np.linalg.eigh(covariances)
        u = w[0] / np.linalg.norm(w[0])
        angle = np.arctan2(u[1], u[0])
        angle = 180 * angle / np.pi  # convert to degrees
        v = 2. * np.sqrt(2.) * np.sqrt(v)
        ell = mpl.patches.Ellipse(gmm.means_[n, :2], v[0], v[1],
                                  180 + angle)#, color=color)
        ell.set_clip_box(ax.bbox)
        ell.set_alpha(0.5)
        ax.add_artist(ell)
        ax.set_aspect('equal', 'datalim')

def gmm_covariance(dataset,labels):
    lowest_bic = np.infty
    bic = []
    n_components_range = range(1, 7)
    cv_types = ['spherical', 'tied', 'diag', 'full']
    for cv_type in cv_types:
        for n_components in n_components_range:
            # Fit a Gaussian mixture with EM
            #for name in dataset:
            for label in labels:
                tr_data, te_data = load_datasets(data_path, name)
                sub_train_data = tr_data[np.where(tr_data[:, 2] == label)]
                gmm = mixture.GaussianMixture(n_components=n_components,
                                              covariance_type=cv_type)
                gmm.fit(sub_train_data)
                bic.append(gmm.bic(sub_train_data))
                if bic[-1] < lowest_bic:
                    lowest_bic = bic[-1]
                    best_gmm = gmm

    bic = np.array(bic)
    color_iter = itertools.cycle(['navy', 'turquoise', 'cornflowerblue',
                                  'darkorange'])
    clf = best_gmm
    bars = []

    plt.figure(figsize=(8, 6))
    spl = plt.subplot(2, 1, 1)
    # for i, (cv_type, color) in enumerate(zip(cv_types, color_iter)):
    #     xpos = np.array(n_components_range) + .2 * (i - 2)
    #     bars.append(plt.bar(xpos, bic[i * len(n_components_range):
    #                                   (i + 1) * len(n_components_range)],
    #                         width=.2, color=color))
    # plt.xticks(n_components_range)
    # plt.ylim([bic.min() * 1.01 - .01 * bic.max(), bic.max()])
    # plt.title('BIC score per model')
    # xpos = np.mod(bic.argmin(), len(n_components_range)) + .65 + \
    #        .2 * np.floor(bic.argmin() / len(n_components_range))
    # plt.text(xpos, bic.min() * 0.97 + .03 * bic.max(), '*', fontsize=14)
    # spl.set_xlabel('Number of components')
    # spl.legend([b[0] for b in bars], cv_types)


    # Plot the winner
    fig, ax = plt.subplots(1,1,figsize=(5,5))
    #splot = plt.subplot(2, 1, 2)
    Y_ = clf.predict(sub_train_data)
    covariance_type = clf.covariance_type
    n_component = clf.n_components
    # for i, (mean, cov, color ) in enumerate(zip(clf.means_, clf.covariances_,
    #                                            color_iter)):
    h = plt.subplot(1, 1,1)
    make_ellipses(clf,h )
    plt.scatter(sub_train_data[:, 0], sub_train_data[:, 1])
    plt.xticks(())
    plt.yticks(())
    plt.title('Selected GMM: full model' + str(n_component) + "components")
    plt.subplots_adjust(hspace=.35, bottom=.02)
    plt.show()
    plt.close()

color_iter = itertools.cycle(['navy', 'c', 'cornflowerblue', 'gold',
                              'darkorange'])
def plot_results(X, Y_, means, covariances, label, title):
    #splot = plt.subplot(1, 1, 1 + index)
    colors = ['steelblue', 'orange']
    markers = ['o','x']
    fig, ax = plt.subplots(1,1,figsize=(5,5))
    for i, (mean, covar, color) in enumerate(zip(
            means, covariances, color_iter)):
        v, w = linalg.eigh(covar)
        v = 2. * np.sqrt(2.) * np.sqrt(v)
        u = w[0] / linalg.norm(w[0])
        # as the DP will not use every component it has access to
        # unless it needs it, we shouldn't plot the redundant
        # components.
        if not np.any(Y_ == i):
            continue
        ax.scatter(X[Y_ == i, 0], X[Y_ == i, 1], marker=markers[label], color=colors[label], label=label)

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
    plt.show()
    plt.close()

def plot_line_graph(line_df):
    # Line Graph by matplotlib with wide-form DataFrame
    fig, ax = plt.subplots(1,1,figsize=(10,10))
    ax.plot(line_df.index, line_df.p1test.values, marker='s', color='r', label='p1test' )
    ax.plot(line_df.index, line_df.p1train.values, marker='o', color='#ffa500', label='p1train')
    ax.plot(line_df.index, line_df.p2test.values, marker='*', color='b', label='p2test')
    ax.plot(line_df.index, line_df.p2train.values, marker='+', color='#00bfff', label = 'p2train')

    #xticks(np.arange(0, 1, step=0.2))
    ax.set_xticks(np.arange(line_df.index.values.min(), line_df.index.values.max()+1))
    plt.title('Accuracy', fontsize=20)
    plt.ylabel('Acc', fontsize=14)
    plt.xlabel('n_component', fontsize=14)
    plt.legend(fontsize=12, loc='best')

    plt.show()
    plt.close()




class Config():
    def __init__(self):
        self.n_mixture = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]


if __name__ == "__main__":
    data_path = "./hw1_data/"
    #np_p1_tr_i, np_p1_tr_t, np_p1_te_i, np_p1_te_t = load_datasets(data_path)

    dataset = ['p1', 'p2', 'all']
    # Plotting train dataset
    for name in dataset:
        tr_data, te_data = load_datasets(data_path, name)
        dataset_plot(name + " train data", tr_data)
    dataset.remove('all')

    # the number of component
    # gmm ploting, p1
    GMM_model = {}
    #n_mixture = 5
    labels = [0,1]
    parameter = Config()
    # line_graph_dataset = gmm_training_plotting(GMM_model, dataset, labels, parameter)
    # line_df = pd.DataFrame(line_graph_dataset)
    # line_df.set_index('n_component', inplace=True)
    # #line_df.set_index('n_component',)
    # plot_line_graph(line_df)


    gmm_covariance(dataset,labels)
    print("done")





    # #gmm training
    # for name in dataset:
    #     for label in labels:
    #         GMM_model[name + "_" + str(label)] = define_GMM(n_mixture)
    #         tr_data, te_data = load_datasets(data_path, name)
    #         sub_train_data = tr_data[np.where(tr_data[:,2]==label)]
    #         GMM_model[name + "_" + str(label)].fit(sub_train_data[:,0:2])
    #
    # #gmm plotting
    # for name in dataset:
    #     for label in labels:
    #         tr_data, te_data = load_datasets(data_path, name)
    #         sub_train_data = tr_data[np.where(tr_data[:,2]==label)]
    #         plot_distribution(GMM_model[name + "_" + str(label)], sub_train_data, label, name + "_" + str(label) )
    #
    #     tr_data, te_data = load_datasets(data_path, name)
    #     validate_gmm(GMM_model, te_data, name, "test")
    #     validate_gmm(GMM_model, tr_data, name, "train")




