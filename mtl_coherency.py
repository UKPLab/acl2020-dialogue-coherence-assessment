import os
import operator
import random
import datetime
import logging
import sys
import argparse
import numpy as np
import pandas as pd
from copy import deepcopy
from collections import Counter
from ast import literal_eval
from tqdm import tqdm, trange
from nltk.corpus import stopwords
from sklearn.metrics import mean_squared_error, f1_score, accuracy_score, label_ranking_average_precision_score, confusion_matrix, average_precision_score

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchtext as tt
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset, Dataset)
from torch.nn.modules.distance import CosineSimilarity
from torch.nn.modules import HingeEmbeddingLoss

from model.mtl_models import CosineCoherence, MTL_Model3
from data_preparation import get_dataloader

test_amount = 1

def main():
    args = parse_args()
    init_logging(args)
    # Init randomization
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    if args.cuda != -1:
        cuda_device_name = "cuda:{}".format(args.cuda)
        device = torch.device(cuda_device_name if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cpu') # if torch.cuda.is_available() else 'cpu')

    logging.info("Used Device = {}".format(str(device)))

    train_datasetfile = os.path.join(args.datadir,"train", "coherency_dset_{}.txt".format(str(args.task)))
    val_datasetfile = os.path.join(args.datadir, "validation", "coherency_dset_{}.txt".format(str(args.task)))
    test_datasetfile = os.path.join(args.datadir, "test", "coherency_dset_{}.txt".format(str(args.task)))

    if args.model == "cosine":
        if args.do_train:
            logging.info("cannot train the cosine model!, we ignore --do_train ")
            args.do_train = False
        model = CosineCoherence(args, device).to(device)
    elif args.model == "random":
        if args.do_train:
            logging.info("cannot train the random model!, we ignore --do_train ")
            args.do_train = False
        model = None
    elif args.model == "model-3":
        model = MTL_Model3(args, device).to(device)
    else:
        raise NotImplementedError("specified model is not supported")

    logging.info("Used Model = {}".format(str(args.model)))
    best_epoch = -1
    train_dl = None
    val_dl = None
    test_dl = None

    if args.do_train:
        logging.info('load training data from: %s'%train_datasetfile)
        train_dl = get_dataloader(train_datasetfile, args)
        logging.info('load validation data from: %s' % train_datasetfile)
        val_dl = get_dataloader(val_datasetfile, args)

        sigma_1 = nn.Parameter(torch.tensor(args.mtl_sigma, requires_grad=True).to(device))
        sigma_2 = nn.Parameter(torch.tensor(args.mtl_sigma, requires_grad=True).to(device))

        if args.loss == "mtl":
            optimizer = torch.optim.Adam(list(model.parameters())+[
                sigma_1,sigma_2], lr=args.learning_rate)
        else:
            optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

        hinge = torch.nn.MarginRankingLoss(reduction='none', margin=0.1).to(device)
        epoch_scores = dict()

        for epoch in trange(args.epochs, desc="Epoch"):
            output_model_file_epoch = os.path.join(args.datadir, "{}_task-{}_loss-{}_epoch-{}.ckpt".format(str(model), str(args.task),str(args.loss), str(epoch)))

            for i,((utts_left, utts_right), 
                    (coh_ixs, (acts_left, acts_right)), (len_u1, len_u2, len_d1, len_d2)) in tqdm(enumerate(train_dl),
                    total=len(train_dl), desc='Training', postfix="LR: {}".format(args.learning_rate)):
                if args.test and i >= test_amount: break

                coh_ixs = coh_ixs.to(device)
                coh1, (da1,loss1) = model(utts_left.to(device),
                        acts_left.to(device),
                        (len_u1.to(device), len_d1.to(device)))
                coh2, (da2,loss2) = model(utts_right.to(device),
                        acts_right.to(device), 
                        (len_u2.to(device), len_d2.to(device)))

                # coh_ixs is of the form [0,1,1,0,1], where 0 indicates the first one is the more coherent one
                # for this loss, the input is expected as [1,-1,-1,1,-1], where 1 indicates the first to be coherent, while -1 the second
                # therefore, we need to transform the coh_ixs accordingly
                loss_coh_ixs = torch.add(torch.add(coh_ixs*(-1), torch.ones(coh_ixs.size()).to(device))*2, torch.ones(coh_ixs.size()).to(device)*(-1))
                loss_da = loss1+loss2
                loss_coh = hinge(coh1, coh2, loss_coh_ixs)
                if args.loss == "da":
                    loss = loss_da
                elif args.loss == "coh":
                    loss = hinge(coh1, coh2, loss_coh_ixs)
                elif args.loss == "mtl":
                    loss = torch.div(loss_da, sigma_1**2) + torch.div(loss_coh, sigma_2**2) + torch.log(sigma_1) + torch.log(sigma_2)
                elif args.loss == 'coin':
                    d = random.uniform(0,1)
                    if d < 0.5:
                        loss = loss_da
                    else:
                        loss = loss_coh
                elif args.loss == 'sum':
                    loss = loss_da + loss_coh

                optimizer.zero_grad()
                loss.mean().backward()
                optimizer.step()

            #save after every epoch
            torch.save(model.state_dict(), output_model_file_epoch)

    if args.do_eval:

        if train_dl == None:
            train_dl = get_dataloader(train_datasetfile, args)
        if val_dl == None:
            val_dl = get_dataloader(val_datasetfile, args)
        test_dl = get_dataloader(test_datasetfile, args)

        def _eval_datasource(dl, desc_str):
            coh_y_true = []
            coh_y_pred = []
            da_f1_scores = []
            da_y_true = []
            da_y_pred = []

            for i,((utts_left, utts_right), 
                    (coh_ixs, (acts_left, acts_right)),
                   (len_u1, len_u2, len_d1, len_d2)) in tqdm(enumerate(dl),
                    total=len(dl), desc=desc_str, postfix="LR: {}".format(args.learning_rate)):

                if args.test and i >= test_amount:
                    break

                if model == None: #generate random values
                    pred = [random.randint(0,1) for _ in range(coh_ixs.size(0))]
                    coh_y_pred += pred
                    coh_y_true += coh_ixs.detach().cpu().numpy().tolist()
                else:

                    coh1, lda1 = model(utts_left.to(device), acts_left.to(device), (len_u1.to(device), len_d1.to(device)))
                    coh2, lda2 = model(utts_right.to(device), acts_right.to(device), (len_u2.to(device), len_d2.to(device)))

                    _, pred = torch.max(torch.cat([coh1.unsqueeze(1), coh2.unsqueeze(1)], dim=1), dim=1)
                    coh_y_pred += pred.detach().cpu().numpy().tolist()
                    coh_y_true += coh_ixs.detach().cpu().numpy().tolist()

                    if lda1 != None and lda2 != None:
                        da1 = lda1[0].view(acts_left.size(0)*acts_left.size(1)).detach().cpu().numpy()
                        da2 = lda2[0].view(acts_left.size(0)*acts_left.size(1)).detach().cpu().numpy()
                        acts_left = acts_left.view(acts_left.size(0)*acts_left.size(1)).detach().cpu().numpy()
                        acts_right = acts_right.view(acts_right.size(0)*acts_right.size(1)).detach().cpu().numpy()
                        acts_left, da1 = da_filter_zero(acts_left.tolist(), da1.tolist())
                        acts_right, da2 = da_filter_zero(acts_right.tolist(), da2.tolist())
                        da_y_pred += da1 + da2
                        da_y_true += acts_left + acts_right

            return (coh_y_true, coh_y_pred), (da_y_true, da_y_pred)

        def _log_dataset_scores(name, coh_y_true, coh_y_pred, da_y_true, da_y_pred):
            logging.info("%s size: %d"%(name, len(coh_y_true) ))
            coh_acc = accuracy_score(coh_y_true, coh_y_pred)
            logging.info("%s coherence accuracy: %2.2f"%(name, coh_acc*100))
            da_acc = accuracy_score(da_y_true, da_y_pred)
            logging.info("%s DA accuracy: %2.2f"%(name, da_acc*100))
            da_f1 = f1_score(da_y_true, da_y_pred, average='weighted')
            logging.info("%s DA MicroF1: %2.2f"%(name, da_f1*100))

        # choose the best epoch
        if args.model != "random" and args.model != "cosine" and args.oot_model == None:
            best_epoch = 0
            best_coh_acc, best_da_acc = None, None
            for i in range(args.epochs):
                model_file_epoch = os.path.join(args.datadir, "{}_task-{}_loss-{}_epoch-{}.ckpt".format(
                    str(model), str(args.task),str(args.loss), str(i)))
                model.load_state_dict(torch.load(model_file_epoch))
                model.to(device)
                model.eval()

                (coh_y_true, coh_y_pred), (da_y_true, da_y_pred) = _eval_datasource(val_dl, "Validation {}:".format(i))
                if i == 0:
                    best_coh_acc = accuracy_score(coh_y_true, coh_y_pred)
                    best_da_acc = accuracy_score(da_y_true, da_y_pred)
                elif args.loss == 'da':
                    curr_da_acc = accuracy_score(da_y_true, da_y_pred)
                    if curr_da_acc > best_da_acc:
                        best_epoch = i
                elif args.loss == 'coh':
                    curr_coh_acc = accuracy_score(coh_y_true, coh_y_pred)
                    if curr_coh_acc > best_coh_acc:
                        best_epoch = i
                elif args.loss == 'mtl' or args.loss == 'coin' or args.loss == 'sum':
                    curr_coh_acc = accuracy_score(coh_y_true, coh_y_pred)
                    curr_da_acc = accuracy_score(da_y_true, da_y_pred)
                    if curr_coh_acc+curr_da_acc > best_coh_acc+best_da_acc:
                        best_epoch = i

            logging.info("Best Epoch = {}".format(best_epoch))
            # evaluate all sets on the best epoch
            model_file_epoch = os.path.join(args.datadir, "{}_task-{}_loss-{}_epoch-{}.ckpt".format(
                str(model), str(args.task),str(args.loss), str(best_epoch)))
            model.load_state_dict(torch.load(model_file_epoch))
            model.to(device)
            model.eval()

        elif args.oot_model:
            model.load_state_dict(torch.load(args.oot_model))
            model.to(device)
            model.eval()

        datasets = [('train', train_dl), ('validation', val_dl), ('test', test_dl)]
        for (name, dl) in datasets:
            (coh_y_true, coh_y_pred), (da_y_true, da_y_pred) = _eval_datasource(dl, "Final Eval {}".format(name))
            _log_dataset_scores(name, coh_y_true, coh_y_pred, da_y_true, da_y_pred)

def da_filter_zero(y_true, y_pred):
    x = zip(y_true, y_pred)
    x = list(filter(lambda y: y[0] != 0, x))
    return [yt for (yt,_) in x], [yp for (_,yp) in x]

def init_logging(args):
    now = datetime.datetime.now()
    proc = "train" if args.do_train else "eval"
    if os.path.exists(args.logdir)== False:
        os.makedirs(args.logdir)
    logfile = os.path.join(args.logdir, 'coherency_{}_{}_loss_{}_task_{}_{}.log'.format(proc, args.model,args.loss ,args.task, now.strftime("%Y-%m-%d-%H-%M-%S")))
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(filename=logfile, filemode='w', level=logging.DEBUG, format='%(levelname)s:%(message)s')
    print("Logging to ", logfile)

    logging.info("Used Hyperparameters:")

    logging.info("learning_rate = {}".format(args.learning_rate))
    logging.info("num_epochs = {}".format(args.epochs))
    logging.info("lstm_hidden_sent = {}".format(args.lstm_sent_size))
    logging.info("lstm_hidden_utt = {}".format(args.lstm_utt_size))
    logging.info("lstm_layers = {}".format(args.lstm_layers))
    logging.info("batch_size = {}".format(args.batch_size))
    logging.info("dropout probability = {}".format(args.dropout_prob))
    logging.info("MTL Sigma Init = {}".format(args.mtl_sigma))
    if args.oot_model:
        logging.info("using OOT Model = {}".format(args.oot_model))
    logging.info("========================")
    logging.info("task = {}".format(args.task))
    logging.info("loss = {}".format(args.loss))
    logging.info("seed = {}".format(args.seed))
    logging.info("embedding = {}".format(args.embedding))
    logging.info("========================")
def parse_args():
    parser = argparse.ArgumentParser()
    # This also serves as a kind of configuration object, so some parameters are not ought to be changed (listed below)
    parser.add_argument("--datadir",
                        required=True,
                        type=str,
                        help="""The input directory where the files of daily
                        dialog are located. the folder should have
                        train/test/validation as subfolders""")
    parser.add_argument("--logdir",
                        default="./logs",
                        type=str,
                        help="the folder to save the logfile to.")
    parser.add_argument('--seed',
                        type=int,
                        default=80591,
                        help="random seed for initialization")
    parser.add_argument('--batch_size',
                        type=int,
                        default=128,
                        help="")
    parser.add_argument('--epochs',
                        type=int,
                        default=20,
                        help="amount of epochs")
    parser.add_argument('--learning_rate',
                        type=float,
                        default=0.0005,
                        help="")
    parser.add_argument('--dropout_prob',
                        type=float,
                        default=0.1,
                        help="the dropout probality for DA classification")
    parser.add_argument('--lstm_sent_size',
                        type=int,
                        default=128,
                        help="hidden size for the lstm models")
    parser.add_argument('--lstm_utt_size',
                        type=int,
                        default=256,
                        help="hidden size for the lstm models")
    parser.add_argument('--mtl_sigma',
                        type=float,
                        default=2.0,
                        help="initialization value for the two sigma values when using MTL Loss")
    parser.add_argument('--embedding',
                        type=str,
                        default="glove",
                        help="""from which embedding should the word ids be used.
                                alternatives: glove """)
    parser.add_argument('--model',
                        type=str,
                        default="cosine",
                        help="""with which model the dataset should be trained/evaluated.
                                alternatives: random | cosine | model-3 """)
    parser.add_argument('--loss',
                        type=str,
                        default="mtl",
                        help="""with which loss the dataset should be trained/evaluated.
                                alternatives: mtl | coin | da | coh """)
    parser.add_argument('--task',
                        required=True,
                        type=str,
                        default="up",
                        help="""for which task the dataset should be created.
                                alternatives: up (utterance permutation)
                                              us (utterance sampling)
                                              ui (utterance insertion)
                                              hup (half utterance petrurbation) """)
    parser.add_argument('--oot_model',
                        required=False,
                        type=str,
                        default=None,
                        help="""when doing Out-Of-Task evaluations, this provides the model file""")
    parser.add_argument('--best_epoch',
                        type=int,
                        default = None,
                        help= "when evaluating, tell the best epoch to choose the file")
    parser.add_argument('--test',
                        action='store_true',
                        help= "just do a test run on small amount of data")
    parser.add_argument('--cuda',
                        type=int,
                        default = -1,
                        help= "which cuda device to take")
    parser.add_argument('--do_train',
                        action='store_true',
                        help= "just do a test run on small amount of data")
    parser.add_argument('--do_eval',
                        action='store_true',
                        help= "just do a test run on small amount of data")
    ### usually unmodified parameters, keept here to have a config like object
    parser.add_argument('--num_classes',
                        type=int,
                        default=0,
                        help="amount of classes 1-4 for DA acts, 0 for none. default: 5 (for DailyDialog), set to 50 for Switchboard")
    parser.add_argument('--lstm_layers',
                        type=int,
                        default=1,
                        help="DONT CHANGE. amount of layers for LSTM models")
    parser.add_argument('--embedding_dim',
                        type=int,
                        default=300,
                        help="DONT CHANGE. embedding dimension for GloVe vectors")

    args = parser.parse_args()
    if args.num_classes == 0:
        dataset = os.path.split(args.datadir)[-1]
        if dataset == "daily_dialog":
            args.num_classes = 5
        else:
            args.num_classes = 50

    return args


##########################################
import time
if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    logging.info("Took %5.2f seconds" % (end_time - start_time))
    logging.shutdown()
