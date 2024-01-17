import argparse
import os
import pickle
import sys

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import grb.utils as utils
from grb.dataset import Dataset
from grb.evaluator import AttackEvaluator

import config_cora as config

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Reproducing results on leaderboards')
    parser.add_argument("--gpu", type=int, default=0, help="gpu")
    parser.add_argument("--dataset", type=str, default="grb-cora")
    parser.add_argument("--dataset_mode", nargs='+', default="full")
    parser.add_argument("--feat_norm", type=str, default="arctan")
    parser.add_argument("--data_dir", type=str, default="../data/")
    parser.add_argument("--model", nargs='+', default=None)
    parser.add_argument("--model_sur", nargs='+', default=None)
    parser.add_argument("--model_dir", type=str, default="../saved_models/")
    parser.add_argument("--model_file", type=str, default="model_0.pt")
    parser.add_argument("--n_attack", type=int, default=0)
    parser.add_argument("--attack", nargs='+', default="fgsm")
    parser.add_argument("--attack_mode", type=str, default="injection")
    parser.add_argument("--attack_dir", type=str, default="../attack_results/")
    parser.add_argument("--attack_adj_name", type=str, default="adj.pkl")
    parser.add_argument("--attack_feat_name", type=str, default="features.npy")
    parser.add_argument("--weight_type", type=str, default="polynomial",
                        help="Type of weighted accuracy, 'polynomial' or 'arithmetic'.")
    parser.add_argument("--save_dir", type=str, default=None)

    args = parser.parse_args()

    if args.gpu >= 0:
        device = "cuda:{}".format(args.gpu)
    else:
        device = "cpu"

    if args.dataset not in args.data_dir:
        args.data_dir = os.path.join(args.data_dir, args.dataset)
    if args.dataset not in args.model_dir:
        args.model_dir = os.path.join(args.model_dir, args.dataset)
    if args.dataset not in args.attack_dir:
        args.attack_dir = os.path.join(args.attack_dir, args.dataset)

    if args.attack is not None:
        args.attack_list = args.attack
    else:
        if args.attack_mode == "modification":
            args.attack_list = config.modification_attack_list
        elif args.attack_mode == "injection":
            args.attack_list = config.injection_attack_list
        else:
            args.attack_list = config.attack_list

    result_dict = {"no_attack": {}}
    if args.attack_dir:
        for attack_name in args.attack_list:
            result_dict[attack_name] = {}

    dataset = Dataset(name=args.dataset,
                      data_dir=args.data_dir,
                      mode=args.dataset_mode,
                      feat_norm=args.feat_norm,
                      verbose=True)

    adj = dataset.adj
    features = dataset.features
    labels = dataset.labels
    num_nodes = dataset.num_nodes
    num_features = dataset.num_features
    num_classes = dataset.num_classes
    train_mask = dataset.train_mask
    val_mask = dataset.val_mask
    test_mask = dataset.test_mask

    if args.model is not None:
        model_list = args.model
    else:
        model_list = config.standard_train_model_list
    if args.model_sur is not None:
        model_sur_list = args.model_sur
    else:
        model_sur_list = config.model_sur_list

    attack_dict = {}
    for i in range(6):
        attack_dict[i] = os.path.join(args.attack_dir, args.attack + "_vs_" + "gcn_ln", str(i))

    model_dict = {}
    for model_name in model_list:
        # Corresponding model path
        if "_at" in model_name:
            model_dict[model_name] = os.path.join(args.model_dir, model_name, "final_model_at_0.pt")
        else:
            model_dict[model_name] = os.path.join(args.model_dir, model_name, args.model_file)

    for model_name in model_list:
        model = torch.load(model_dict[model_name])
        model = model.to(device)
        model.eval()

        test_score = utils.evaluate(model,
                                    features=features,
                                    adj=adj,
                                    labels=dataset.labels,
                                    adj_norm_func=model.adj_norm_func,
                                    mask=dataset.test_mask,
                                    device=device)
        print("Test score after attack for target model: {:.4f}.".format(test_score) + "rate0")
        for i in range(1, 6):
            with open(os.path.join(attack_dict[i], args.attack_adj_name), 'rb') as f:
                adj_attack = pickle.load(f)
            adj_attack = sp.csr_matrix(adj_attack)
            adj_attacked = sp.vstack([adj, adj_attack[:, :num_nodes]])
            adj_attacked = sp.hstack([adj_attacked, adj_attack.T])
            adj_attacked = sp.csr_matrix(adj_attacked)
            features_attack = np.load(os.path.join(attack_dict[i], args.attack_feat_name))
            features_attacked = np.concatenate([features, features_attack])
            features_attacked = utils.feat_preprocess(features=features_attacked, device=device)

            test_score = utils.evaluate(model,
                                        features=features_attacked,
                                        adj=adj_attacked,
                                        labels=dataset.labels,
                                        adj_norm_func=model.adj_norm_func,
                                        mask=dataset.test_mask,
                                        device=device)
            print("Test score after attack for target model: {:.4f}.".format(test_score) + "rate" + str(i))

            del adj_attack, adj_attacked, features_attacked
        print(model_name)
        print("******************************************")
