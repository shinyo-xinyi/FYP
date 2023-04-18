'''
    Test the single attack influence on the voice spoofing detection system using the evaluation data set.
'''

import argparse
import torch
import eval_metrics as em
import numpy as np
import matplotlib.pyplot as plt
import os

def test_individual_attacks(cm_score_file):
    # Load CM scores
    cm_data = np.genfromtxt(cm_score_file, dtype=str)
    cm_utt_id = cm_data[:, 0]
    cm_sources = cm_data[:, 1]
    cm_keys = cm_data[:, 2]
    cm_scores = cm_data[:, 3].astype(np.float)

    eer_cm_lst, min_tDCF_lst = [], []
    for attack_idx in range(7,20):
        print('attack type:', attack_idx)

        # Extract bona fide (real human) and spoof scores from the CM scores
        bona_cm = cm_scores[cm_keys == 'bonafide']
        spoof_cm = cm_scores[cm_sources == 'A%02d' % attack_idx]

        eercm, threshold = em.compute_eer(bona_cm, spoof_cm)
        print('eer:', eercm)
        print('threshold:', threshold)

        # Visualize CM scores
        plt.figure()
        plt.hist(bona_cm, histtype='step', density=True, bins=50, label='Bona fide')
        plt.hist(spoof_cm, histtype='step', density=True, bins=50, label='Spoof')
        plt.plot(threshold, 0, 'o', markersize=10, mfc='none', mew=2, clip_on=False, label='EER threshold')
        plt.text(threshold,1,threshold, color = 'green')
        plt.legend()
        plt.xlabel('CM score')
        plt.ylabel('Density')
        plt.title('CM score histogram (Attack_A%02d)'% attack_idx)
        plt.savefig(cm_score_file[:-23] + args.loss + '_' + '_A%02d.png' % attack_idx)
        eer_cm_lst.append(eercm)
    return eer_cm_lst

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-m', '--model_dir', type=str, help="path to the trained model", default="./models/softmax/40epochs")
    parser.add_argument('-l', '--loss', type=str, default="softmax",
                        choices=["softmax", 'amsoftmax', 'ocsoftmax'], help="loss function")
    parser.add_argument("--gpu", type=str, help="GPU index", default="0")
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    eer_cm_lst = test_individual_attacks(os.path.join(args.model_dir, 'eval_cm_score.txt'))


