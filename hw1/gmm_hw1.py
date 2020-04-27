
import sklearn.mixture as mixture
import numpy as np
import librosa
import pdb
import glob
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from mpl_toolkits.mplot3d import Axes3D
import matplotlib as cm


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


def plot_distribution(model, X_train):
    x = np.linspace(-100., 100.)
    y = np.linspace(-100., 100.)  # 간격을 나눠주고
    X, Y = np.meshgrid(x, y)  # mesh로 그림을 그리고
    XX = np.array([X.ravel(), Y.ravel()]).T
    Z = model.score_samples(XX)
    Z = Z.reshape(X.shape)
    CS = plt.contour(X, Y, -np.log10(-Z), antialiased=False)  # contour 그래프
    CB = plt.colorbar(CS)
    plt.scatter(X_train[:, 0], X_train[:, 1], .8)
    plt.title('Negative log-likelihood predicted by a GMM')
    plt.axis('tight')
    plt.show()
    plt.close()

    fig = plt.gca(projection='3d')
    surf = fig.plot_surface(X, Y, -np.log10(-Z), cmap='coolwarm', linewidth=0, antialiased=False)
    plt.colorbar(surf)
    plt.title('3D contour of GMM')
    fig.set_xlabel('x')
    fig.set_ylabel('y')
    fig.set_zlabel('z')
    plt.show()
    plt.close()

def load_datasets(data_path, train_label):

    np_p1_tr_i = np.loadtxt(data_path + "/p1_train_input.txt")
    np_p1_tr_t = np.loadtxt(data_path + "/p1_train_target.txt")
    np_p1_te_i = np.loadtxt(data_path + "/p1_test_input.txt")
    np_p1_te_t = np.loadtxt(data_path + "/p1_test_target.txt")

    np_p1_tr_a = np.column_stack([np_p1_tr_i, np_p1_tr_t])
    np_p1_tr_a_x = np_p1_tr_a[np.where(np_p1_tr_a[:,2] == train_label)]
    #np_p1_tr_a_1 = np_p1_tr_a[np.where(np_p1_tr_a[:, 2] == 1)]
    return np_p1_tr_a_x[:,0:2]

def load_test_data(data_path):

    #np_p1_tr_i = np.loadtxt(data_path + "/p1_train_input.txt")
    #np_p1_tr_t = np.loadtxt(data_path + "/p1_train_target.txt")
    np_p1_te_i = np.loadtxt(data_path + "/p1_test_input.txt")
    np_p1_te_t = np.loadtxt(data_path + "/p1_test_target.txt")

    np_p1_te_a = np.column_stack([np_p1_te_i, np_p1_te_t])
    #np_p1_te_a_x = np_p1_te_a[np.where(np_p1_te_a[:,2] == train_label)]
    #np_p1_tr_a_1 = np_p1_tr_a[np.where(np_p1_tr_a[:, 2] == 1)]
    return np_p1_te_a[:,0:2], np_p1_te_a[:,2]

def define_GMM(n_mixture):
    return mixture.GaussianMixture(n_components=n_mixture, covariance_type='diag')


if __name__ == "__main__":
    data_path = "./hw1_data/"
    #np_p1_tr_i, np_p1_tr_t, np_p1_te_i, np_p1_te_t = load_datasets(data_path)
    GMM_model = {}
    n_mixture = 10
    train_label = [0,1]
    for i in train_label:
        GMM_model[i] = define_GMM(n_mixture)
        dataset = load_datasets(data_path, i)
        GMM_model[i].fit(dataset)

    ACC = 0
    test_data, test_label = load_test_data(data_path)
    for i, data in enumerate(test_data):
        data = np.expand_dims(data, axis=0)
        candidates = []
        # Calculate likelihood scores for all the trained GMMs.
        for label in GMM_model.keys():
            score = GMM_model[label].score(data)
            print("label {} : {}".format(label, score))
            candidates.append(score)
        candidates = np.array(candidates)

        estimated_speaker_label = np.argmax(candidates)
        print("Estimated: {}, True: {}".format(estimated_speaker_label, test_label[i]), end=' ' * 5)
        if test_label[i] == estimated_speaker_label:
            print("correct!")
            ACC += 1
        else:
            print("incorrect...")
    print("ACC:{:.2f}".format(ACC / len(test_label) * 100.))



    define_GMM
    # # toy project
    # n_mixture = 2
    # toy_data = generate_examples(n_mixture)
    # gmm_model = define_GMM(n_mixture)
    # gmm_model.fit(toy_data)  # GMM의 평균과 코베리언스가 나타남
    # plot_distribution(gmm_model, toy_data)



