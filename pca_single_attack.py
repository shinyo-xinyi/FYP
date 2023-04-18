import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import argparse
import os

'''PCA'''


def pca(X, y, model_dir):
    pca = PCA(n_components=2)
    reduced_x = pca.fit_transform(X)

    x_list = [[], [], [], [], [], [], [], [], [], [], [], [], [], []]
    y_list = [[], [], [], [], [], [], [], [], [], [], [], [], [], []]

    label_list = ['A07', 'A08', 'A09', 'A10', 'A11', 'A12', 'A13', 'A14', 'A15', 'A16', 'A17', 'A18', 'A19', 'Bonafide']
    color_list = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#bcbd22', '#17becf',
                  '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#bec1d4', '#bb7784', '#0000ff', '#111010', '#FFFF00',
                  '#1f77b4', '#800080', '#959595',
                  '#7d87b9', '#bec1d4', '#d6bcc0', '#bb7784', '#8e063b']

    for i in range(len(reduced_x)):
        if y[i] == 'A07':
            x_list[0].append(reduced_x[i][0])
            y_list[0].append(reduced_x[i][1])
        elif y[i] == 'A08':
            x_list[1].append(reduced_x[i][0])
            y_list[1].append(reduced_x[i][1])
        elif y[i] == 'A09':
            x_list[2].append(reduced_x[i][0])
            y_list[2].append(reduced_x[i][1])
        elif y[i] == 'A10':
            x_list[3].append(reduced_x[i][0])
            y_list[3].append(reduced_x[i][1])
        elif y[i] == 'A11':
            x_list[4].append(reduced_x[i][0])
            y_list[4].append(reduced_x[i][1])
        elif y[i] == 'A12':
            x_list[5].append(reduced_x[i][0])
            y_list[5].append(reduced_x[i][1])
        elif y[i] == 'A13':
            x_list[6].append(reduced_x[i][0])
            y_list[6].append(reduced_x[i][1])
        elif y[i] == 'A14':
            x_list[7].append(reduced_x[i][0])
            y_list[7].append(reduced_x[i][1])
        elif y[i] == 'A15':
            x_list[8].append(reduced_x[i][0])
            y_list[8].append(reduced_x[i][1])
        elif y[i] == 'A16':
            x_list[9].append(reduced_x[i][0])
            y_list[9].append(reduced_x[i][1])
        elif y[i] == 'A17':
            x_list[10].append(reduced_x[i][0])
            y_list[10].append(reduced_x[i][1])
        elif y[i] == 'A18':
            x_list[11].append(reduced_x[i][0])
            y_list[11].append(reduced_x[i][1])
        elif y[i] == 'A19':
            x_list[12].append(reduced_x[i][0])
            y_list[12].append(reduced_x[i][1])
        elif y[i] == 'Bonafide':
            x_list[13].append(reduced_x[i][0])
            y_list[13].append(reduced_x[i][1])

    font = {'family': 'Times New Roman',
            'size': 16,
            }
    sns.set(font_scale=1.2)

    plt.rc('font', family='Times New Roman')
    '''
    for i in range(0,13):
        plt.scatter(x_list[i], y_list[i], c=color_list[i], s = 50, alpha = 0.3,marker='o', label=label_list[i])
        plt.legend()
        save_path = os.path.join(model_dir, 'pca_single_attack_'+str(i))
        plt.savefig(save_path, dpi=120)
        plt.show()
    '''
    # plt.title("PCA analysis (single attack)")
    for i in range(0, 13):
        plt.figure()
        plt.scatter(x_list[13], y_list[13], c=color_list[13], s=50, alpha=0.5, marker='x', label=label_list[13])
        plt.scatter(x_list[i], y_list[i], c=color_list[i], s=50, alpha=0.5, marker='o', label=label_list[i])
        plt.legend()
        save_path = os.path.join(model_dir, 'pca_single_attack_' + str(i))
        plt.savefig(save_path, dpi=120)
        # plt.show()

    print(pca.explained_variance_ratio_)


if __name__ == "__main__":
    model_dir = './models/ocsoftmax/20epochs/'
    cm_score_file = os.path.join(model_dir, 'checkpoint_cm_score.txt')
    cm_data = np.genfromtxt(cm_score_file, dtype=str)
    cm_sources = cm_data[:, 1]
    cm_keys = cm_data[:, 2]
    cm_scores = cm_data[:, 3].astype(np.float)

    labels = np.load(os.path.join(model_dir, 'labels2.npy'))
    feats = np.load(os.path.join(model_dir, 'embeddings2.npy'))
    attack_labels = np.array([])
    attack_feats = np.zeros(shape=(0, 256))

    for attack_idx in range(7, 20):
        current_feats = feats[cm_sources == 'A%02d' % attack_idx]
        attack_feats = np.concatenate((attack_feats, current_feats))
        # print("num of attack 'A%02d':"% attack_idx, len(attack_feats))
        current_label = np.array('A%02d' % attack_idx)
        attack_labels = np.concatenate((attack_labels, np.tile(current_label, len(current_feats))))

    current_feats = feats[cm_sources == 'A00']
    attack_feats = np.concatenate((attack_feats, current_feats))
    current_label = np.array('Bonafide')
    attack_labels = np.concatenate((attack_labels, np.tile(current_label, len(current_feats))))

    pca(attack_feats, attack_labels, model_dir)