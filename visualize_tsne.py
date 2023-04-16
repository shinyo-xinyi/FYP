'''
    Visualize the extracted embeddings distribution using T-SNE method.
'''

import matplotlib.pyplot as plt
from sklearn import manifold
import numpy as np

def t_sne(X,y):
    tsne = manifold.TSNE(n_components=2, init='pca', random_state=501)
    X_tsne = tsne.fit_transform(X)
    print("Org data dimension is {}. Embedded data dimension is {}".format(X.shape[-1], X_tsne.shape[-1]))

    x_min, x_max = X_tsne.min(0), X_tsne.max(0)
    X_norm = (X_tsne - x_min) / (x_max - x_min)  # 归一化
    plt.figure(figsize=(8, 8))
    for i in range(X_norm.shape[0]):
        plt.text(X_norm[i, 0], X_norm[i, 1], 'o', color=plt.cm.Set1(y[i]),
                 fontdict={'weight': 'bold', 'size': 9})
    plt.xticks([])
    plt.yticks([])
    plt.savefig('./models/ocsoftmax/40epochs/tsne_40_eval.png', dpi=120)
    plt.show()

if __name__ == "__main__":
    labels = np.load('./models/ocsoftmax/40epochs/labels2.npy')
    feats = np.load('./models/ocsoftmax/40epochs/embeddings2.npy')
    t_sne(feats, labels)