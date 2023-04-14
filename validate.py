import argparse
import os
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
from dataset import ASVspoof2019
from evaluate_tDCF_asvspoof19 import compute_eer_and_tdcf
from tqdm import tqdm
import eval_metrics as em
import numpy as np
import matplotlib.pyplot as plt
from sklearn import manifold, datasets

def val_model(feat_model_path, loss_model_path, part, add_loss, device):
    embedding_list = []
    label_list = []
    dirname = os.path.dirname
    basename = os.path.splitext(os.path.basename(feat_model_path))[0]
    if "checkpoint" in dirname(feat_model_path):
        dir_path = dirname(dirname(feat_model_path))
    else:
        dir_path = dirname(feat_model_path)
    model = torch.load(feat_model_path, map_location="cuda")
    model = model.to(device)
    loss_model = torch.load(loss_model_path) if add_loss != "softmax" else None
    val_set = ASVspoof2019("LA", "features/ASVspoof2019/LA/Features_Python/",
                            "dataset/LA/LA/ASVspoof2019_LA_cm_protocols/", part,
                            "LFCC", feat_len=750, padding="repeat")
    valDataLoader = DataLoader(val_set, batch_size=16, shuffle=False, num_workers=0,
                                collate_fn=val_set.collate_fn)
    model.eval()

    for i, (lfcc, audio_fn, tags, labels) in enumerate(tqdm(valDataLoader)):
        lfcc = lfcc.unsqueeze(1).float().to(device)
        # tags = tags.to(device)
        labels = labels.to(device)
        feats, lfcc_outputs = model(lfcc)
        # score = F.softmax(lfcc_outputs)[:, 0]

        # print('feats',feats) # 256d
        # print('lfcc_outputs',lfcc_outputs) # 2d

        for j in range(labels.size(0)):
            feat = feats[j].cpu().detach().numpy()
            # print(feat)
            label = labels[j].cpu().detach().numpy()
            # print(label)
            embedding_list.append(feat)
            label_list.append(label)

    embeddings = np.array(embedding_list)
    labels = np.array(label_list)

    np.save("./models1028/ocsoftmax/embeddings.npy", embeddings)
    np.save("./models1028/ocsoftmax/labels.npy", labels)

def val(model_dir, add_loss, device):
    model_path = os.path.join(model_dir, "anti-spoofing_lfcc_model.pt")
    loss_model_path = os.path.join(model_dir, "anti-spoofing_loss_model.pt")
    val_model(model_path, loss_model_path, "dev", add_loss, device)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-m', '--model_dir', type=str, help="path to the trained model", default="./models1028/ocsoftmax/aug/")
    parser.add_argument('-l', '--loss', type=str, default="ocsoftmax",
                        choices=["softmax", 'amsoftmax', 'ocsoftmax'], help="loss function")
    parser.add_argument("--gpu", type=str, help="GPU index", default="0")
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    val(args.model_dir, args.loss, args.device)


