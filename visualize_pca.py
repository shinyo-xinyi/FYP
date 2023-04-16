'''
    Visualize the extracted embeddings distribution using PCA method.
'''

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

'''PCA'''
def pca(X,y):
    pca = PCA(n_components=2)
    reduced_x = pca.fit_transform(X)

    yes_x, yes_y = [], []
    no_x, no_y = [], []

    for i in range(len(reduced_x)):
        if y[i] == 1:
            yes_x.append(reduced_x[i][0])
            yes_y.append(reduced_x[i][1])
        elif y[i] == 0:
            no_x.append(reduced_x[i][0])
            no_y.append(reduced_x[i][1])

    font = {'family': 'Times New Roman',
            'size': 16,
            }
    sns.set(font_scale=1.2)

    plt.rc('font', family='Times New Roman')
    plt.scatter(yes_x, yes_y, c='b', marker='o', label='spoof')
    plt.scatter(no_x, no_y, c='r', marker='x', label='bonafide')
    plt.title("PCA analysis")
    plt.legend()
    plt.savefig('./models1028/ocsoftmax/aug/pca_40_dev.png', dpi=120)
    plt.show()

    print(pca.explained_variance_ratio_)


if __name__ == "__main__":
    labels = np.load('./models1028/ocsoftmax/aug/labels.npy')
    feats = np.load('./models1028/ocsoftmax/aug/embeddings.npy')
    pca(feats, labels)