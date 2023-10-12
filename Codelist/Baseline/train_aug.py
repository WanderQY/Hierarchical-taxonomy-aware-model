#!/usr/bin/env python
# -*- coding: utf-8 -*-
import random
import sys
sys.path.append('../../BirdCLEF/')
import datetime
import os
from mixup import *
from pathlib import Path
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
from torch.optim import lr_scheduler
from tqdm import tqdm
from datetime import datetime
import shutil
import logging
import json
import time
from model import *
from dataloader import BirdsoundData
from evaluation import inference
from utils import parse_datasets
from class_labels import *
# from loss import *
# from augmentation import *
import warnings
warnings.filterwarnings("ignore")


class ModelTrainer:
    def __init__(self, model, train_loader, valid_loader, criterion, optimizer, scheduler, config, num_classes,
                 dataset_name, epochs, save_path, ckpt_path=None, comment=None):
        # Essential parts
        self.device = torch.device('cuda')
        self.model = model.to(self.device)
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.sch_name = config['scheduler']['name']
        self.save_path = save_path
        self.dataset_name = dataset_name
        self.num_classes = num_classes
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.exp_path = Path(os.path.join(save_path, dataset_name, datetime.now().strftime('%Y-%m-%d')))  # 2022-06-09
        self.exp_path.mkdir(exist_ok=True, parents=True)

        # Set logger
        self.logger = logging.getLogger('')
        self.logger.setLevel(logging.INFO)
        fh = logging.FileHandler(os.path.join(self.exp_path, 'training.log'))
        sh = logging.StreamHandler(sys.stdout)
        self.logger.addHandler(fh)
        self.logger.addHandler(sh)

        # Dump hyper-parameters
        # config_info = {'optim':str(self.optimizer), 'scheduler':str(self.scheduler), 'criterion':str(self.criterion)}
        with open(str(self.exp_path.joinpath('config.json')), 'w') as f:
            json.dump(config, f, indent=2) 

        if comment != None:
            self.logger.info(comment)

        self.dataset_name = dataset_name
        self.epochs = epochs
        self.best_mrr = 0.0
        self.best_epoch = 0
        self.flag = 0
        self.total_train_step = 0

        if ckpt_path != None:
            self.load_checkpoint(ckpt_path)


    def train(self):
        train_loss_list, valid_loss_list = [], []
        for epoch in tqdm(range(self.epochs)):
            start = time.time()
            train_loss, t_accuracy = self.train_single_epoch()
            mrr, top_1, top_5, map, valid_loss, _, _ = inference(self.model, self.valid_loader, self.criterion, self.device)
            train_loss_list.append(train_loss)
            valid_loss_list.append(valid_loss)
            duration = time.time() - start

            if mrr > self.best_mrr:
                self.best_mrr = mrr
                self.best_epoch = epoch
                self.flag = 1

            if self.sch_name == 'multistep':
                self.scheduler.step()
            elif self.sch_name == 'plateau':
                self.scheduler.step(valid_loss)

            self.logger.info \
                ("epoch: {} --- t_loss: {:0.3f}, t_acc: {:0.3f}%, v_loss: {:0.3f}, MRR: {:0.3f}%, MAP: {:0.3f}%"
                 " v_top1: {:0.3f}%, v_top5: {:0.3f}%" \
                 .format(epoch + 1, train_loss, t_accuracy, valid_loss, mrr, map, top_1, top_5))
            self.logger.info \
                ("------------- best_mrr: {:0.3f}%, best_epoch: {}, cost_time: {:0.2f}s, update_step: {}" \
                 .format(self.best_mrr, self.best_epoch + 1, duration, self.total_train_step))

            if self.flag == 1:
                self.save_checkpoint(epoch, mrr, map, top_1, top_5, True)
                self.flag = 0
            else:
                self.save_checkpoint(epoch, mrr, map, top_1, top_5, False)

        # plot
        epoch_list = [i + 1 for i in range(self.epochs)]
        plt.plot(epoch_list, train_loss_list)
        plt.plot(epoch_list, valid_loss_list)
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.legend(["Train_loss", "Valid_loss"], loc='upper right')
        plt.savefig("{}/train-valid_loss.png".format(self.exp_path), format='png')
        plt.show()
        return self.best_epoch, self.best_mrr

    def train_single_epoch(self):
        self.model.train()
        total_loss = 0.0
        batch_corr, total = 0, 0
        batch_size = len(self.train_loader)
        for b, batch in enumerate(self.train_loader):
            self.total_train_step += 1
            _, feats, labels, _ = batch
            total += labels.size(0)
            # B, C, H, W = images.shape
            feats = feats.to(self.device)
            labels = labels.to(self.device)
            if random.random() <= 0.75:
                feats, labels_a, labels_b, lam = mixup_data(feats, labels)
                # forward
                outputs = self.model(feats)
                loss_func = mixup_criterion(labels_a, labels_b, lam)
                batch_loss = loss_func(self.criterion, outputs)
            else:
                outputs = self.model(feats)
                batch_loss = self.criterion(outputs, labels)

            self.optimizer.zero_grad()
            # backward
            batch_loss.backward()
            # update parameters
            self.optimizer.step()
            batch_corr += (outputs.argmax(1) == labels).sum()
            total_loss += batch_loss.item()
            print("{}/{} --- {}".format(b, batch_size, batch_loss.detach()), end='\r')
        total_loss = total_loss / len(self.train_loader)
        ACC = float(batch_corr) / total * 100
        return total_loss, ACC

    def load_checkpoint(self, ckpt):
        self.logger.info(f"Loading checkpoint from {ckpt}")
        checkpoint = torch.load(os.path.join(self.save_path, self.dataset_name) + '/{}'.format(ckpt), map_location=self.device)
        pretrained_dict = checkpoint['model_state_dict']
        optimizer_params = checkpoint['optimizer']
        print(optimizer_params)
        self.model.load_state_dict(pretrained_dict)
        self.optimizer.load_state_dict(optimizer_params)

    def save_checkpoint(self, epoch, mrr, map, top_1, top_5, best=True):
        state_dict = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'MRR': mrr,
            'MAP': map,
            'TOP1_accuracy': top_1,
            'TOP5_accuracy': top_5
        }

        self.exp_path.joinpath('ckpt').mkdir(exist_ok=True, parents=True)
        save_path = "{}/ckpt/last.pt".format(self.exp_path)
        torch.save(state_dict, save_path)
        if best:
            shutil.copyfile(save_path, '{}/ckpt/best.pt'.format(self.exp_path))


if __name__ == '__main__':
    class_list = SELECT_CLASS
    num_classes = len(class_list)
    save_path = sys.path[-1] + 'Results/'
    dataset_name = 'Baseline'
    model = Xception(num_classes=num_classes, use_attention=True)
    criterion = nn.CrossEntropyLoss()
    config = {'optim': {'name': 'Adam',
                        'lr': 1e-2},
              'scheduler': {'name': 'plateau',
                            'param': [0.1, 5]},
              'criterion': {'name': 'CrossEntropyLoss'}}
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=1e-5, eps=1e-3)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, min_lr=1e-4)
    epochs = 200
    #split_datas = json.load(open(sys.path[-1] + 'SplitDatas/split_dataset1.json'))
    split_datas = json.load(open(sys.path[-1] + 'SplitDatas/small_dataset1.json'))
    train_datas, valid_datas = parse_datasets(split_datas, add_birdsonly=True, replace_dir=False)
    train_dataset = BirdsoundData(train_datas, option='train', class_list=class_list, augment=['noise', 'distortion', 'cut_mix'])
    valid_dataset = BirdsoundData(valid_datas, option='test', class_list=class_list)
    del split_datas, train_datas, valid_datas
    train_loader = DataLoader(train_dataset, batch_size=32, num_workers=2, shuffle=True, drop_last=True, pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size=1, num_workers=0, drop_last=True, pin_memory=True)
    del train_dataset, valid_dataset
    # ckpt_path = '2023-02-09/ckpt/best.pt'
    Train = ModelTrainer(model, train_loader, valid_loader, criterion, optimizer, scheduler, config, num_classes, dataset_name,
                         epochs, save_path)
    best_epoch, best_mrr = Train.train()
