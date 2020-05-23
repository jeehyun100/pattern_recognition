
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.preprocessing import OneHotEncoder
import os
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
import pandas as pd
import scipy


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

    return np_p_tr_a[:,0:2],tr_target, np_p_te_a[:,0:2],te_target


def grid_svm(data_path, dataset):
    data_name = 'p1'
    for data_name in dataset:
        row_dict = dict()
        #param_grid = {'C': [0.1, 1, 10, 100, 500], 'gamma': [500, 100, 10, 1, 0.1, 0.01, 0.001], 'kernel': ['rbf']}
        tr_data, tr_target, te_data, te_target = load_datasets(data_path, data_name)
        param_grid = {'C': scipy.stats.expon(scale=100), 'gamma': scipy.stats.expon(scale=100),
         'kernel': ['rbf'], 'class_weight': ['balanced', None]}

        X = tr_data
        Y = tr_target
        grid = RandomizedSearchCV(svm.SVC(), param_grid, refit=True, verbose=2)
        grid.fit(X, Y)
        best_c = grid.best_estimator_.C
        best_gamma = grid.best_estimator_.gamma
        best_param = "rbf_" + '{:.2f}'.format(best_c) +"_" + '{:.2f}'.format(best_gamma) + "_"+ data_name
        title, acc = plot_graph_save_csv(grid.best_estimator_, te_data, te_target, best_param)
        row_dict[title] = acc
        df_result = pd.DataFrame.from_dict(row_dict, orient='index')
        df_result = df_result.T
        df_result.to_csv("./csv/svm_kernel_" + title + "_" + data_name + ".csv")
        print(df_result)
        print(grid.best_estimator_)

        target_prime2 = grid.predict(te_data)
        accuracy = (target_prime2 == te_target).mean()


def kernel_svm_c_gamma(data_path, dataset, c, gamma ):
    #data_name = 'p2'
    for data_name in dataset:
        tr_data, tr_target, te_data, te_target = load_datasets(data_path, data_name)

        X = tr_data
        Y = tr_target
        # fit the model

        row_dict = dict()
        kernel = 'rbf'

        clf = svm.SVC(kernel=kernel, gamma=gamma, C=c)
        clf.fit(X, Y)
        train_target_prime = clf.predict(X)
        tr_accuracy = (train_target_prime == tr_target).mean()
        type = str(c) + "_" + str(gamma)
        title = kernel + "_" + data_name + "_"+type
        title, acc = plot_graph_save_csv(clf, te_data, te_target, title)
        row_dict[title] = acc
        df_result = pd.DataFrame.from_dict(row_dict, orient='index')
        df_result = df_result.T

        df_result.to_csv("./csv/svm_kernel_"+ data_name + "_"+type + ".csv")
        print(df_result)
        return tr_accuracy, acc

def kernel_svm(data_path, dataset ):
    """ SVM kernel change

    """
    #data_name = 'p2'
    for data_name in dataset:
        tr_data, tr_target, te_data, te_target = load_datasets(data_path, data_name)

        X = tr_data
        Y = tr_target
        # fit the model
        #result_list = list()
        row_dict = dict()
        for kernel in ('linear', 'poly', 'rbf'):

            clf = svm.SVC(kernel=kernel, gamma=2, C=10)
            clf.fit(X, Y)
            train_target_prime = clf.predict(X)
            tr_accuracy = (train_target_prime == tr_target).mean()
            title = kernel + "_" + data_name
            title, acc = plot_graph_save_csv(clf, te_data, te_target, title)
            #row_dict[title] = acc
            row_dict[title] = acc
            #result_list.extend(dict(acc))
        df_result = pd.DataFrame.from_dict(row_dict, orient='index')
        df_result = df_result.T
        df_result.to_csv("./csv/svm_kernel_"+ data_name + ".csv")
        print(df_result)


def plot_graph_save_csv(clf, te_data, te_target, title ):
    results = dict()
    fig, ax = plt.subplots(1,1,figsize=(4,3))

    ax.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=40,
                facecolors='none', zorder=10, edgecolors='k')

    target_prime = clf.predict(te_data)
    accuracy = (target_prime == te_target).mean() * 100
    results[title] = accuracy

    x_min, x_max = te_data[:, 0].min() - 0.2, te_data[:, 0].max() + 0.2
    y_min, y_max = te_data[:, 1].min() - 0.2, te_data[:, 1].max() + 0.2

    XX, YY = np.mgrid[x_min:x_max:200j, y_min:y_max:200j]
    Z = clf.decision_function(np.c_[XX.ravel(), YY.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(XX.shape)

    ax.pcolormesh(XX, YY, Z > 0, cmap=plt.cm.Paired)
    ax.contour(XX, YY, Z, colors=['k', 'k', 'k'], linestyles=['--', '-', '--'],
                levels=[-.5, 0, .5])

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    te_target_r = te_target.reshape(-1, 1)
    data = np.hstack((te_data, te_target_r))

    labels = [0,1]
    colors = ['steelblue', 'orange']
    markers = ['o','x']
    for i , label in enumerate(labels):
        X = data[np.where(data[:,2]==label)][:,0]
        Y = data[np.where(data[:,2]==label)][:,1]

        ax.scatter(X, Y, marker=markers[i], color=colors[i], label=label)
    plt.title(title + "(Acc:{0:.2f}%)".format(accuracy))
    plt.savefig("./plot/svm_"+ title + "(Acc:{0:.2f}%)".format(accuracy)+".png")
    plt.show()
    return title, accuracy


def graph_svm_with_param(data_path):
    best_param_c_p1 = [18.86, 2, 2, 1, 3, 61.55]#, 10, 15, 20, 30, 40, 50, 60, 70, 80, 90]
    best_param_gamma_p1 = [14.93, 3, 2, 1, 3, 13.62]#, 10, 18, 25, 30, 40, 50, 60, 70, 80, 90]
    for c, g in zip(best_param_c_p1, best_param_gamma_p1):
        tr_accuracy, acc = kernel_svm_c_gamma(data_path, ['p1'], c, g)
        print("panalty : {0} gamma : {1} train acc {2} test acc{3} ".format(c,g,tr_accuracy, acc))


def overgitting_svm_with_param(data_path, datasets):
    best_param_c_p1 = [1, 5, 10, 15, 20, 30, 40, 50, 60, 70, 80, 90]
    best_param_gamma_p1 = [1, 5, 10, 18, 25, 30, 40, 50, 60, 70, 80, 90]

    for data_name in datasets:
        result_list = list()
        for c, g in zip(best_param_c_p1, best_param_gamma_p1):
            row_dict = dict()
            tr_acc, te_acc = kernel_svm_c_gamma(data_path, [data_name], c, g)
            row_dict['tr_acc'] = tr_acc
            row_dict['te_acc'] = te_acc
            result_list.append(row_dict)
        df_plot = pd.DataFrame(result_list, index = best_param_c_p1)
        df_plot.plot()
        plt.xlabel('Panalty parameter')
        plt.ylabel('Acc')
        plt.xticks(best_param_c_p1)
        plt.title("SVM Overffiting " + str(data_name))
        plt.savefig("./plot/svm_overfitting_"+ data_name + ".png")
        plt.show()
    print(df_plot)


if __name__ == "__main__":
    os.makedirs("./ckpt/", exist_ok=True)
    os.makedirs("./csv/", exist_ok=True)
    os.makedirs("./plot/", exist_ok=True)

    # load dataset
    data_path = "../hw2_data"
    dataset = ['p1', 'p2']
    # figure number
    fignum = 1
    # load dataset
    data_path = "./hw2_data"
    dataset = ['p1', 'p2']

    # kernel function change
    #kernel_svm(data_path,dataset )

    # grid search svm
    #grid_svm(data_path,dataset )

    # overfitting test
    #overgitting_svm_with_param(data_path, dataset)

    # Best svm plot
    graph_svm_with_param(data_path)

