import argparse
import os
import torch
from torch.utils.data import DataLoader
from dataset import ASVspoof2019
from tqdm import tqdm
import numpy as np

def val_model(feat_model_path, loss_model_path, part, add_loss, device):
    embedding_list = []
    label_list = []
    model = torch.load(feat_model_path, map_location="cuda")
    model = model.to(device)
    val_set = ASVspoof2019("LA", "features/ASVspoof2019/LA/Features_Python/",
                            "dataset/LA/LA/ASVspoof2019_LA_cm_protocols/", part,
                            "LFCC", feat_len=750, padding="repeat")
    valDataLoader = DataLoader(val_set, batch_size=16, shuffle=False, num_workers=0,
                                collate_fn=val_set.collate_fn)
    model.eval()
    for i, (lfcc, audio_fn, tags, labels) in enumerate(tqdm(valDataLoader)):
        lfcc = lfcc.unsqueeze(1).float().to(device)
        labels = labels.to(device)
        feats, lfcc_outputs = model(lfcc)
        for j in range(labels.size(0)):
            feat = feats[j].cpu().detach().numpy()
            label = labels[j].cpu().detach().numpy()
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


