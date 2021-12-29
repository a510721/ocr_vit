import os
import logging
import numpy as np
from collections import OrderedDict

import torch
import torch.nn.functional as F
from torch.nn.utils.clip_grad import clip_grad_norm_

from tqdm import *
from src.utils.utils import AverageMeter, Eval, OCRLabelConverter

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class OCRTrainer(object):
    def __init__(self, opt):
        super(OCRTrainer, self).__init__()
        self.data_train = opt.data_train
        self.data_val = opt.data_val
        self.model = opt.model
        self.criterion = opt.criterion
        self.optimizer = opt.optimizer
        self.schedule = opt.schedule
        self.converter = OCRLabelConverter(opt.alphabet)
        self.evaluator = Eval()
        print('Scheduling is {}'.format(self.schedule))

        self.batch_size = opt.batch_size
        self.count = opt.epoch
        self.epochs = opt.epochs
        self.cuda = opt.cuda
        self.collate_fn = opt.collate_fn
        self.init_meters()


    def init_meters(self):
        self.avgTrainLoss = AverageMeter("Train loss")
        self.avgTrainCharAccuracy = AverageMeter("Train Character Accuracy")
        self.avgTrainWordAccuracy = AverageMeter("Train Word Accuracy")
        self.avgValLoss = AverageMeter("Validation loss")
        self.avgValCharAccuracy = AverageMeter("Validation Character Accuracy")
        self.avgValWordAccuracy = AverageMeter("Validation Word Accuracy")


    def forward(self, x):
        logits = self.model(x)
        return logits.transpose(1,0)

    def loss_fn(self, logits, targets, pred_sizes, target_sizes):
        loss = self.criterion(logits, targets, pred_sizes, target_sizes)
        return loss

    def step(self):
        self.max_grad_norm = 1.0 #0.01
        clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        self.optimizer.step()

    def schedule_lr(self):
        if self.schedule:
            self.schedule.step()

    def _run_batch(self, batch, report_accuracy=False, validation=False):
        input_, targets = batch['img'].to(device), batch['label']
        targets, lengths = self.converter.encode(targets) #charcter -> index digits

        logits = self.forward(input_)
        logits = logits.contiguous().cpu()
        logits = torch.nn.functional.log_softmax(logits,2) # [T(length),B(Batch), H(Class)] : class probabilty
        T, B, H = logits.size()

        pred_sizes = torch.LongTensor( [T for i in range(B)])
        targets = targets.view(-1).contiguous()
        loss = self.loss_fn(logits, targets, pred_sizes, lengths)

        if report_accuracy:
            probs, preds = logits.max(2)
            preds = preds.transpose(1, 0).contiguous().view(-1)
            sim_preds = self.converter.decode(preds.data, pred_sizes.data, raw=False)
            ca = np.mean((list(map(self.evaluator.char_accuracy, list(zip(sim_preds, batch['label']))))))
            wa = np.mean((list(map(self.evaluator.word_accuracy, list(zip(sim_preds, batch['label']))))))
        return loss, ca, wa

    def run_epoch(self, validation=False):
        if not validation:
            loader = self.train_dataloader()
            pbar = tqdm(loader, desc='Epoch: [%d]/[%d] Training'%(self.count,self.epochs), leave=True)
            self.model.train()
        else:
            loader = self.val_dataloader()
            pbar = tqdm(loader, desc='Validating', leave=True)
            self.model.eval()
        outputs = []

        for batch_nb, batch in enumerate(pbar):
            if not validation:
                output = self.training_step(batch)
            else:
                output = self.validation_step(batch)
            pbar.set_postfix(output)
            outputs.append(output)

        self.schedule_lr()

        if not validation:
            result = self.train_end(outputs)
        else:
            result = self.validation_end(outputs)
        return result

    def training_step(self, batch):
        #print(batch['img'].shape)
        loss, ca, wa = self._run_batch(batch, report_accuracy=True)
        #zero gradient -> backward loss -> optimize step(prevent gradnorm)
        self.optimizer.zero_grad()
        loss.backward()
        self.step()
        output = OrderedDict({
            'loss': abs(loss.item()),
            'train_ca': ca.item(),
            'train_wa': wa.item()
            })
        return output

    def validation_step(self, batch):
        loss, ca, wa = self._run_batch(batch, report_accuracy=True, validation=True)
        output = OrderedDict({
            'val_loss': abs(loss.item()),
            'val_ca': ca.item(),
            'val_wa': wa.item()
            })
        return output

    def train_dataloader(self):
        # logging.info('training data loader called')
        loader = torch.utils.data.DataLoader(self.data_train,
                                             batch_size=self.batch_size,
                                             collate_fn=self.collate_fn,
                                             shuffle=True)
        return loader

    def val_dataloader(self):
        # logging.info('val data loader called')
        loader = torch.utils.data.DataLoader(self.data_val,
                                             batch_size=self.batch_size,
                                             collate_fn=self.collate_fn)
        return loader


    def train_end(self, outputs):
        for output in outputs:
            self.avgTrainLoss.add(output['loss'])
            self.avgTrainCharAccuracy.add(output['train_ca'])
            self.avgTrainWordAccuracy.add(output['train_wa'])

        train_loss_mean = abs(self.avgTrainLoss.compute())
        train_ca_mean = self.avgTrainCharAccuracy.compute()
        train_wa_mean = self.avgTrainWordAccuracy.compute()

        result = {'train_loss': train_loss_mean, 'train_ca': train_ca_mean, 'train_wa': train_wa_mean}
        # result = {'progress_bar': tqdm_dict, 'log': tqdm_dict, 'val_loss': train_loss_mean}
        return result

    def validation_end(self, outputs):
        for output in outputs:
            self.avgValLoss.add(output['val_loss'])
            self.avgValCharAccuracy.add(output['val_ca'])
            self.avgValWordAccuracy.add(output['val_wa'])

        val_loss_mean = abs(self.avgValLoss.compute())
        val_ca_mean = self.avgValCharAccuracy.compute()
        val_wa_mean = self.avgValWordAccuracy.compute()

        result = {'val_loss': val_loss_mean, 'val_ca': val_ca_mean, 'val_wa': val_wa_mean}
        return result





'''
if __name__ == '__main__':
    from src.options.opts import base_opts
    from argparse import ArgumentParser
    parser = ArgumentParser()
    base_opts(parser)
    args = parser.parse_args()

    trainer = OCRTrainer(args)

'''