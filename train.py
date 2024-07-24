import os
import torch
from glob import glob
from tkinter import Image
import cv2
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from tqdm import tqdm
from DDPM import TrainDDPM as DDPM
from MambaDDPM import TrainMambaDDPM as MambaDDPM
from MambaDDPMBSA import TrainMambaDDPMBSA as MambaDDPMBSA
import torchvision.transforms as transforms
import argparse


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', type=str, default="DDPM", choices=['DDPM', 'MambaDDPM', 'MambaDDPMBSA'],
                        help='The model to use: DDPM, MambaDDPM, or MambaDDPMBSA')
    parser.add_argument('--state', type=str, default="train", choices=['train', 'eval', 'sample'],
                        help='The state of the model: train, eval, or sample')
    parser.add_argument('--epoch', type=int, default=200, help='Number of epochs for training')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--T', type=int, default=1000, help='Number of timesteps')
    parser.add_argument('--channel', type=int, help='Number of channels')
    parser.add_argument('--channel_mult', type=list, help='Channel multipliers')
    parser.add_argument('--num_res_blocks', type=int, help='Number of residual blocks')
    parser.add_argument('--dropout', type=float, help='Dropout rate')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--multiplier', type=float, default=2.5, help='Multiplier for loss scaling')
    parser.add_argument('--beta_1', type=float, default=1e-4, help='Beta 1 for the Adam optimizer')
    parser.add_argument('--beta_T', type=float, default=0.028, help='Beta T for the Adam optimizer')
    parser.add_argument('--img_size', type=int, default=224, help='Image size')
    parser.add_argument('--grad_clip', type=float, default=1.0, help='Gradient clipping value')
    parser.add_argument('--device', type=str, default="cuda:0", help='Device to use for training/testing')
    parser.add_argument('--w', type=float, default=1.8, help='Weight parameter')
    parser.add_argument('--data_dir', type=str, default="./Data/HAM10000", help='Directory to dataset')
    parser.add_argument('--save_dir', type=str, default="./CheckpointsCondition/", help='Directory to save checkpoints')
    parser.add_argument('--training_load_weight', type=str, default=None, help='Path to training load weights')
    parser.add_argument('--test_load_weight', type=str, default=None, help='Path to test load weights')
    parser.add_argument('--sampled_dir', type=str, default="./SampledImgs/", help='Directory to save sampled images')
    parser.add_argument('--sampledNoisyImgName', type=str, default="NoisyGuidenceImgs.png",
                        help='Filename for noisy guidance images')
    parser.add_argument('--sampledImgName', type=str, default="SampledGuidenceImgs.jpg",
                        help='Filename for sampled guidance images')
    parser.add_argument('--nrow', type=int, help='Number of rows for sampled image grid')
    parser.add_argument('--numOfClass', type=list, help='Number of sampled image class')

    return parser.parse_args()

class HAM10000(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        # Load data and get label
        X = Image.open(self.df['path'][index])
        y = torch.tensor(int(self.df['cell_type_idx'][index]))
        if self.transform:
            X = self.transform(X)
        return X, y
def compute_img_mean_std(image_paths):
    img_h, img_w = 224, 224
    imgs = []
    means, stdevs = [], []

    for i in tqdm(range(len(image_paths))):
        img = cv2.imread(image_paths[i])
        img = cv2.resize(img, (img_h, img_w))
        imgs.append(img)

    imgs = np.stack(imgs, axis=3)
    print(imgs.shape)

    imgs = imgs.astype(np.float32) / 255.

    for i in range(3):
        pixels = imgs[:, :, i, :].ravel()  # resize to one row
        means.append(np.mean(pixels))
        stdevs.append(np.std(pixels))

    means.reverse()  # BGR --> RGB
    stdevs.reverse()

    print("normMean = {}".format(means))
    print("normStd = {}".format(stdevs))
    return means, stdevs

def main(args,dataset):

    if args.model == "DDPM":
        if args.state == "train":
            return DDPM.train(args, dataset)
        if args.state == "sample":
            return DDPM.sample(args)
        if args.state == "eval":
            return DDPM.eval(args)
        else:
            return print("state error!")
    if args.model == "MambaDDPM":
        if args.state == "train":
            return MambaDDPM.train(args, dataset)
        if args.state == "sample":
            return MambaDDPM.sample(args)
        if args.state == "eval":
            return MambaDDPM.eval(args)
        else:
            return print("state error!")
    if args.model == "MambaDDPMBSA":
        if args.state == "sample":
            return MambaDDPMBSA.sample(args)
        if args.state == "train":
            return MambaDDPMBSA.train(args,dataset)
        if args.state == "eval":
            return MambaDDPMBSA.eval(args)
        else:
            return print("state error!")


if __name__ == '__main__':
    args = parse_args()

    np.random.seed(777)
    torch.manual_seed(777)
    torch.cuda.manual_seed(777)

    data_dir = args.data_dir
    all_image_path = glob(os.path.join(data_dir, '*', '*.jpg'))
    imageid_path_dict = {os.path.splitext(os.path.basename(x))[0]: x for x in all_image_path}
    lesion_type_dict = {
        'nv': 'Melanocytic nevi',
        'mel': 'dermatofibroma',
        'bkl': 'Benign keratosis-like lesions ',
        'bcc': 'Basal cell carcinoma',
        'akiec': 'Actinic keratoses',
        'vasc': 'Vascular lesions',
        'df': 'Dermatofibroma'
    }
    # HAM10000 mean std
    # norm_mean = [0.7630331, 0.5456457, 0.5700467]
    # norm_std = [0.1409281, 0.15261227, 0.16997086]
    norm_mean, norm_std = compute_img_mean_std(all_image_path)
    df_original = pd.read_csv(os.path.join(data_dir, 'HAM10000_metadata.csv'))
    df_original['path'] = df_original['image_id'].map(imageid_path_dict.get)
    df_original['cell_type'] = df_original['dx'].map(lesion_type_dict.get)
    df_original['cell_type_idx'] = pd.Categorical(df_original['cell_type']).codes

    train_transform = transforms.Compose(
        [transforms.Resize((args.img_size, args.img_size)), transforms.RandomHorizontalFlip(),
         transforms.RandomVerticalFlip(), transforms.RandomRotation(20),
         transforms.ColorJitter(brightness=0.1, contrast=0.1, hue=0.1),
         transforms.ToTensor(), transforms.Normalize(norm_mean, norm_std)])
    dataset = HAM10000(df_original, transform=train_transform)
    # train
    main(dataset)
