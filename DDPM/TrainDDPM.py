import os
from datetime import datetime
from typing import Dict
import numpy as np

import torch
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image

from MambaDDPM.MambaDiffusion import GaussianDiffusionSampler, GaussianDiffusionTrainer
from Scheduler import GradualWarmupScheduler
from Model import UNet

def train(cfg, dataset):
    device = torch.device(cfg.device)
    # dataset
    dataloader = DataLoader(
        dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=4, drop_last=True, pin_memory=True)

    # model setup
    model = UNet(T=cfg.T, num_labels=10, ch=cfg.channel,
                 ch_mult=cfg.channel_mult,
                 num_res_blocks=cfg.num_res_blocks, dropout=cfg.dropout).to(device)
    if cfg.training_load_weight is not None:
        model.load_state_dict(torch.load(os.path.join(
            cfg.save_dir, cfg.training_load_weight), map_location=device), strict=False)
        print("Model weight load down.")
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=1e-4)
    cosineScheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer, T_max=cfg.epoch, eta_min=0, last_epoch=-1)
    warmUpScheduler = GradualWarmupScheduler(optimizer=optimizer, multiplier=cfg.multiplier,
                                             warm_epoch=cfg.epoch // 10, after_scheduler=cosineScheduler)
    trainer = GaussianDiffusionTrainer(
        model, cfg.beta_1, cfg.beta_T, cfg.T).to(device)

    # start training
    for e in range(cfg.epoch):
        with tqdm(dataloader, dynamic_ncols=True) as tqdmDataLoader:
            for images, labels in tqdmDataLoader:
                # train
                b = images.shape[0]
                optimizer.zero_grad()
                x_0 = images.to(device)
                labels = labels.to(device) + 1
                if np.random.rand() < 0.1:
                    labels = torch.zeros_like(labels).to(device)
                loss = trainer(x_0, labels).sum() / b ** 2.
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), cfg.grad_clip)
                optimizer.step()
                tqdmDataLoader.set_postfix(ordered_dict={
                    "epoch": e,
                    "loss: ": loss.item(),
                    "img shape: ": x_0.shape,
                    "LR": optimizer.state_dict()['param_groups'][0]["lr"]
                })
        warmUpScheduler.step()
        torch.save(model.state_dict(), os.path.join(
            cfg.save_dir, 'ckpt_' + str(e) + "_.pt"))


def eval(cfg):
    device = torch.device(cfg.device)
    # load model and evaluate
    with torch.no_grad():
        step = int(cfg.batch_size // 10)
        labelList = []
        k = 0
        for i in range(1, cfg.batch_size + 1):
            labelList.append(torch.ones(size=[1]).long() * k)
            if i % step == 0:
                if k < 10 - 1:
                    k += 1
        labels = torch.cat(labelList, dim=0).long().to(device) + 1
        print("labels: ", labels)
        model = UNet(T=cfg.T, num_labels=10, ch=cfg.channel,
                     ch_mult=cfg.channel_mult,
                     num_res_blocks=cfg.num_res_blocks, dropout=cfg.dropout).to(device)
        ckpt = torch.load(os.path.join(
            cfg.save_dir, cfg.test_load_weight), map_location=device)
        model.load_state_dict(ckpt)
        print("model load weight done.")
        model.eval()
        sampler = GaussianDiffusionSampler(
            model, cfg.beta_1, cfg.beta_T, cfg.T, w=cfg.w).to(device)
        # Sampled from standard normal distribution
        noisyImage = torch.randn(
            size=[cfg.batch_size, 3, cfg.img_size, cfg.img_size], device=device)
        saveNoisy = torch.clamp(noisyImage * 0.5 + 0.5, 0, 1)
        save_image(saveNoisy, os.path.join(
            cfg.sampled_dir, cfg.sampledNoisyImgName), nrow=cfg.nrow)
        sampledImgs = sampler(noisyImage, labels)
        sampledImgs = sampledImgs * 0.5 + 0.5  # [0 ~ 1]
        print(sampledImgs)
        save_image(sampledImgs, os.path.join(
            cfg.sampled_dir, cfg.sampledImgName), nrow=cfg.nrow)


def sample(cfg):
    device = torch.device(cfg.device)
    strTime = datetime.now().strftime("%Y-%m-%d-%H-%M-")
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],  # ImageNet 数据集的均值
            std=[0.229, 0.224, 0.225]  # ImageNet 数据集的标准差
        )
    ])

    with torch.no_grad():
        labelList = []
        numOfClass = cfg.numOfClass
        for i in range(len(numOfClass)):
            num_elements = numOfClass[i].item()
            elements = torch.full((num_elements,), i)
            labelList.append(elements)

        labels = torch.cat(labelList, dim=0).long().to(device) + 1
        print("labels: ", labels)
        model = UNet(T=cfg.T, num_labels=10, ch=cfg.channel,
                     ch_mult=cfg.channel_mult,
                     num_res_blocks=cfg.num_res_blocks, dropout=cfg.dropout).to(device)
        ckpt = torch.load(os.path.join(
            cfg.save_dir, cfg.test_load_weight), map_location=device)
        model.load_state_dict(ckpt)
        print("model load weight done.")
        model.eval()
        sampler = GaussianDiffusionSampler(
            model, cfg.beta_1, cfg.beta_T, cfg.T, w=cfg.w).to(device)

        # begin synthetic
        noisyImage = torch.randn(
            size=[cfg.batch_size, 3, cfg.img_size, cfg.img_size], device=device)
        sampledImgs = sampler(noisyImage, labels)
        sampledImgs = sampledImgs * 0.5 + 0.5  # [0 ~ 1]
        synthetic_data_preprocessed = torch.stack([transform(img) for img in sampledImgs])

        save_image(synthetic_data_preprocessed, os.path.join(
            cfg.sampled_dir, strTime + cfg.sampledImgName),
                   nrow=cfg.nrow)

