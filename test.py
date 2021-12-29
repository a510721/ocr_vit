import sys
import os
import cv2
import pdb
import json
import math
import pickle
import logging
import warnings
from tqdm import *
import numpy as np
import torch
from torch import nn
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision
from torch.utils.data import random_split
from argparse import ArgumentParser

from src.options.opts import base_opts
from src.criterions.ctc import CustomCTCLoss
from src.utils.utils import *
from src.models.crnn import CRNN
from src.data.pickle_dataset import PickleDataset
from src.data.synth_dataset import SynthDataset, SynthCollator

warnings.filterwarnings("ignore")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0)
np.random.seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def get_accuracy(args):
    loader = torch.utils.data.DataLoader(args.data,batch_size=args.batch_size,collate_fn=args.collate_fn)
    model = args.model
    model.eval()
    converter = OCRLabelConverter(args.alphabet)
    evaluator = Eval()
    labels, predictions = [], []
    for iteration, batch in enumerate(tqdm(loader)):
        input_, target = batch['img'].to(device), batch['label']
        labels.extend(target)
        targets, lengths = converter.encode(target)
        logits = model(input_).transpose(1,0) # [batch, length, class] -> [ length, batch, class]
        logits = torch.nn.functional.log_softmax(logits, 2) # value -> probability
        logits = logits.contiguous().cpu()

        T, B, H = logits.size()#[T:34, B:32 H:81]
        pred_sizes = torch.LongTensor([T for i in range(B)])
        probs, pos = logits.max(2) #class, [ max_probability, pos(index) ]
        pos = pos.transpose(1, 0).contiguous().view(-1) #[34(length), 32(batch)] -> [32(batch), 34(length)] ->[1088]
        sim_preds = converter.decode(pos.data, pred_sizes.data, raw=False) #[1088], [32]
        predictions.extend(sim_preds)
    ca = np.mean((list(map(evaluator.char_accuracy, list(zip(predictions, labels))))))
    wa = np.nanmean((list(map(evaluator.word_accuracy_line, list(zip(predictions, labels))))))

    return ca, wa


def main(**kwargs):
    parser = ArgumentParser()
    base_opts(parser)
    args = parser.parse_args()
    args.data = SynthDataset(args)
    args.collate_fn = SynthCollator()
    args.alphabet = """Only thewigsofrcvdampbkuq.$A-210xT5'MDL,RYHJ"ISPWENj&BC93VGFKz();#:!7U64Q8?+*ZX/%"""
    args.nClasses = len(args.alphabet)

    model = CRNN(args)
    model = model.cuda()
    resume_file = os.path.join(args.save_dir, args.name, 'best.ckpt')

    if os.path.isfile(resume_file):
        print('Loading model %s'%resume_file)
        checkpoint = torch.load(resume_file)
        model.load_state_dict(checkpoint['state_dict'])
        args.model = model
        ca, wa = get_accuracy(args)
        print("Character Accuracy: %.2f\nWord Accuracy: %.2f"%(ca, wa))
    else:
        print("=> no checkpoint found at '{}'".format(save_file))
        print('Exiting')


if __name__ == '__main__':
    main()

