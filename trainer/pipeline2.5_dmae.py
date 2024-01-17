import os
import pickle

import torch
import numpy as np
import torch_geometric
from grb.dataset import Dataset
from torch_geometric.data import Data
from grb.utils import utils
import scipy.sparse as sp
from torch_geometric.utils import to_scipy_sparse_matrix, subgraph

import time
import argparse
import torch_geometric.transforms as T

# custom modules
from dmae.utils import Logger, set_seed, tab_printer
from dmae.model import MaskGAE, DegreeDecoder, EdgeDecoder, GNNEncoder, DotEdgeDecoder
from dmae.mask import MaskEdge, MaskPath


def edge_prediction(splits, batch_size=2 ** 16):
    model.eval()
    train_data = splits['train'].to(device)
    z = model(train_data.x, train_data.edge_index)

    results = model.edge_pred(
        z, splits['test'].pos_edge_label_index, batch_size=batch_size)

    return results


@torch.no_grad()
def graph_reconstruct(predict_edges):
    reconstruct_index = []
    for i in range(predict_edges.size()[0]):
        if predict_edges[i] > 0.35:
            reconstruct_index += [i]
    reconstruct_edges = splits['test'].pos_edge_label_index[:, reconstruct_index]

    return reconstruct_edges


def adj2edge_index(adj_csr):
    adj_nd = np.concatenate(([np.int64(adj_csr.tocoo().col)], [np.int64(adj_csr.tocoo().row)]), 0)
    return torch.from_numpy(adj_nd)


def build_adj(attr, edge_index, adj_type='csr'):
    if type(attr) == torch.Tensor:
        attr = attr.numpy()
    if type(edge_index) == torch.Tensor:
        edge_index = edge_index.numpy()
    if type(edge_index) == tuple:
        edge_index = [edge_index[0].numpy(), edge_index[1].numpy()]
    if adj_type == 'csr':
        adj = sp.csr_matrix((attr, edge_index))
    elif adj_type == 'coo':
        adj = sp.coo_matrix((attr, edge_index))

    return adj


def train_linkpred(model, splits, args, device="cpu"):
    def train(data):
        model.train()
        loss = model.train_epoch(data.to(device), optimizer,
                                 alpha=args.alpha, batch_size=args.batch_size)
        return loss

    @torch.no_grad()
    def test(splits, batch_size=2 ** 16):
        model.eval()
        train_data = splits['train'].to(device)
        z = model(train_data.x, train_data.edge_index)

        valid_auc, valid_ap = model.test(
            z, splits['valid'].pos_edge_label_index, splits['valid'].neg_edge_label_index, batch_size=batch_size)

        test_auc, test_ap = model.test(
            z, splits['test'].pos_edge_label_index, splits['test'].neg_edge_label_index, batch_size=batch_size)

        results = {'AUC': (valid_auc, test_auc), 'AP': (valid_ap, test_ap)}
        return results

    @torch.no_grad()
    def edge_prediction(splits, batch_size=2 ** 16):
        model.eval()
        train_data = splits['train'].to(device)
        z = model(train_data.x, train_data.edge_index)

        results = model.edge_pred(
            z, splits['test'].pos_edge_label_index, batch_size=batch_size)

        return results

    @torch.no_grad()
    def graph_reconstruct(predict_edges):

        reconstruct_index = []
        for i in range(predict_edges.size()[0]):
            if predict_edges[i] > 0.5:
                reconstruct_index += [i]
        reconstruct_edges = splits['test'].pos_edge_label_index[:, reconstruct_index]

        return reconstruct_edges

    monitor = 'AUC'
    save_path = args.save_path
    loggers = {
        'AUC': Logger(args.runs, args),
        'AP': Logger(args.runs, args),
    }
    print('Start Training...')
    for run in range(args.runs):
        model.reset_parameters()

        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=args.lr,
                                     weight_decay=args.weight_decay)

        best_valid = 0.0
        best_epoch = 0
        cnt_wait = 0
        for epoch in range(1, 1 + args.epochs):
            # print("epoch :", epoch)
            t1 = time.time()
            loss = train(splits['train'])
            t2 = time.time()

            if epoch % args.eval_period == 0:
                results = test(splits)

                valid_result = results[monitor][0]
                if valid_result > best_valid:
                    best_valid = valid_result
                    best_epoch = epoch
                    torch.save(model.state_dict(), save_path)
                    cnt_wait = 0
                else:
                    cnt_wait += 1
                if args.debug:
                    for key, result in results.items():
                        valid_result, test_result = result
                        print(key)
                        print(f'Run: {run + 1:02d} / {args.runs:02d}, '
                              f'Epoch: {epoch:02d} / {args.epochs:02d}, '
                              f'Best_epoch: {best_epoch:02d}, '
                              f'Best_valid: {best_valid:.2%}%, '
                              f'Loss: {loss:.4f}, '
                              f'Valid: {valid_result:.2%}, '
                              f'Test: {test_result:.2%}',
                              f'Training Time/epoch: {t2 - t1:.3f}')
                    print('#' * round(140 * epoch / (args.epochs + 1)))
                if cnt_wait == args.patience:
                    print('Early stopping!')
                    break
        print('##### Testing on {}/{}'.format(run + 1, args.runs))

        model.load_state_dict(torch.load(save_path))
        results = test(splits, model)

        for key, result in results.items():
            valid_result, test_result = result
            print(key)
            print(f'**** Testing on Run: {run + 1:02d}, '
                  f'Best Epoch: {best_epoch:02d}, '
                  f'Valid: {valid_result:.2%}, '
                  f'Test: {test_result:.2%}')

        for key, result in results.items():
            loggers[key].add_result(run, result)

    print('##### Final Testing result')
    for key in loggers.keys():
        print(key)
        loggers[key].print_statistics()

    pred_res = edge_prediction(splits, model)
    edge_rec = graph_reconstruct(pred_res)
    # print(pred_res)
    # # print(pred_res.size()[0])
    # print(edge_rec)


parser = argparse.ArgumentParser()
parser.add_argument("--dataset", nargs="?", default="grb-cora", help="Datasets. (default: grb-cora)")
parser.add_argument("--save_dir", type=str, default="../attack_results/")
parser.add_argument("--data_dir", type=str, default="../data/")
parser.add_argument("--attack", nargs='+', default="fgsm")
parser.add_argument("--attack_adj_name", type=str, default="adj.pkl")
parser.add_argument("--attack_feat_name", type=str, default="features.npy")
parser.add_argument("--mask", nargs="?", default="Edge",
                    help="Masking stractegy, `Path`, `Edge` or `None` (default: Path)")
parser.add_argument('--seed', type=int, default=2022, help='Random seed for model and dataset. (default: 2022)')

parser.add_argument('--bn', action='store_true',
                    help='Whether to use batch normalization for GNN encoder. (default: False)')
parser.add_argument("--layer", nargs="?", default="gcn", help="GNN layer, (default: gcn)")
parser.add_argument("--encoder_activation", nargs="?", default="elu",
                    help="Activation function for GNN encoder, (default: elu)")
parser.add_argument('--encoder_channels', type=int, default=128, help='Channels of GNN encoder. (default: 128)')
parser.add_argument('--hidden_channels', type=int, default=128,
                    help='Channels of hidden representation. (default: 128)')
parser.add_argument('--decoder_channels', type=int, default=64, help='Channels of decoder. (default: 64)')
parser.add_argument('--encoder_layers', type=int, default=1, help='Number of layers of encoder. (default: 1)')
parser.add_argument('--decoder_layers', type=int, default=2, help='Number of layers for decoders. (default: 2)')
parser.add_argument('--encoder_dropout', type=float, default=0.8, help='Dropout probability of encoder. (default: 0.7)')
parser.add_argument('--decoder_dropout', type=float, default=0.2, help='Dropout probability of decoder. (default: 0.3)')
parser.add_argument('--alpha', type=float, default=0.003, help='loss weight for degree prediction. (default: 2e-3)')

parser.add_argument('--lr', type=float, default=1e-2, help='Learning rate for training. (default: 1e-2)')
parser.add_argument('--weight_decay', type=float, default=5e-5, help='weight_decay for training. (default: 5e-5)')
parser.add_argument('--grad_norm', type=float, default=1.0, help='grad_norm for training. (default: 1.0.)')
parser.add_argument('--batch_size', type=int, default=2 ** 16, help='Number of batch size. (default: 2**16)')

parser.add_argument("--start", nargs="?", default="edge",
                    help="Which Type to sample starting nodes for random walks, (default: edge)")
parser.add_argument('--p', type=float, default=0.7, help='Mask ratio or sample ratio for MaskEdge/MaskPath')

parser.add_argument('--epochs', type=int, default=500, help='Number of training epochs. (default: 300)')
parser.add_argument('--runs', type=int, default=1, help='Number of runs. (default: 10)')
parser.add_argument('--eval_period', type=int, default=10, help='(default: 10)')
parser.add_argument('--patience', type=int, default=30, help='(default: 30)')
parser.add_argument("--save_path", nargs="?", default="model_linkpred",
                    help="save path for model. (default: model_linkpred)")
parser.add_argument('--debug', action='store_true', help='Whether to log information in each epoch. (default: False)')
parser.add_argument("--device", type=int, default=0)

# ==========================================================

torch.cuda.empty_cache()

try:
    args = parser.parse_args()
    print(tab_printer(args))
except:
    parser.print_help()
    exit(0)

if not args.save_path.endswith('.pth'):
    args.save_path += '.pth'

set_seed(args.seed)
if args.device < 0:
    device = "cpu"
else:
    device = f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"

# ================================================================================
if args.dataset not in args.data_dir:
    args.data_dir = os.path.join(args.data_dir, args.dataset)
if args.dataset not in args.save_dir:
    args.save_dir = os.path.join(args.save_dir, args.dataset)
if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)

dataset = Dataset(name=args.dataset,
                  data_dir=args.data_dir,
                  mode='full',
                  feat_norm='arctan')

adj = dataset.adj
features = dataset.features
labels = dataset.labels
num_features = dataset.num_features
num_classes = dataset.num_classes
test_mask = dataset.test_mask

train_val_mask = torch.logical_or(dataset.train_mask, dataset.val_mask)
train_val_index = torch.where(train_val_mask)[0]
# features = features[train_val_mask]
adj_train = utils.adj_preprocess(adj,
                                 adj_norm_func=None,
                                 mask=train_val_mask,
                                 model_type=torch,
                                 device="cuda:0")

edge_index = adj2edge_index(adj_train)
data_fromGRB = Data(x=features, edge_index=edge_index)

# reconstruct graph==================================================================
transform = T.Compose([
    T.ToUndirected(),
    T.ToDevice(device),
])

data = transform(data_fromGRB)
train_data, val_data, test_data = T.RandomLinkSplit(num_val=0.05, num_test=0.1,
                                                    is_undirected=True,
                                                    split_labels=True,
                                                    add_negative_train_samples=True)(data)

splits = dict(train=train_data, valid=val_data, test=test_data)

if args.mask == 'Path':
    mask = MaskPath(p=args.p, num_nodes=data.num_nodes,
                    start=args.start,
                    walk_length=args.encoder_layers + 1)
elif args.mask == 'Edge':
    mask = MaskEdge(p=args.p)
else:
    mask = None

encoder = GNNEncoder(data.num_features, args.encoder_channels, args.hidden_channels,
                     num_layers=args.encoder_layers, dropout=args.encoder_dropout,
                     bn=args.bn, layer=args.layer, activation=args.encoder_activation)

if args.decoder_layers == 0:
    edge_decoder = DotEdgeDecoder()
else:
    edge_decoder = EdgeDecoder(args.hidden_channels, args.decoder_channels,
                               num_layers=args.decoder_layers, dropout=args.decoder_dropout)

degree_decoder = DegreeDecoder(args.hidden_channels, args.decoder_channels,
                               num_layers=args.decoder_layers, dropout=args.decoder_dropout)

model = MaskGAE(encoder, edge_decoder, degree_decoder, mask).to(device)

print(model)

train_linkpred(model, splits, args, device=device)

# ============================================================================

#
attack_dict = {}
for i in range(6):
    attack_dict[i] = os.path.join(args.save_dir, args.attack + "_vs_" + "gcn_ln", str(i))


for i in range(0, 6):
    if i != 0:
        with open(os.path.join(attack_dict[i], args.attack_adj_name), 'rb') as f:
            adj_attack = pickle.load(f)
        adj_attack = sp.csr_matrix(adj_attack)
        adj_attacked = sp.vstack([adj, adj_attack[:, :dataset.num_nodes]])
        adj_attacked = sp.hstack([adj_attacked, adj_attack.T])
        adj_attacked = sp.csr_matrix(adj_attacked)
        features_attack = np.load(os.path.join(attack_dict[i], args.attack_feat_name))
        features_attacked = np.concatenate([features, features_attack])
    else:
        adj_attacked = adj
        features_attacked = features

    features_attacked = utils.feat_preprocess(features=features_attacked, device=device)
    edge_index_attacked = adj2edge_index(adj_attacked)
    data_attacked = Data(x=features_attacked, edge_index=edge_index_attacked)
    train_data, val_data, test_data = T.RandomLinkSplit(num_val=0.01, num_test=0.05,
                                                        is_undirected=True,
                                                        split_labels=True,
                                                        add_negative_train_samples=True)(data_attacked)
    splits = dict(train=train_data, valid=val_data, test=test_data)
    pred_res = edge_prediction(splits, model)
    edge_rec = graph_reconstruct(pred_res)
    edge_rec = edge_rec.to(device)
    full_edge = torch.cat((train_data.edge_index, edge_rec), 1)
    attr = torch.ones(full_edge[0].shape[0])
    adj_rec = build_adj(attr, full_edge.to('cpu'), adj_type='csr')

    save_dir = os.path.join(args.save_dir, args.attack + "_vs_" + "gcn_ln", "dmae" + str(i))
    utils.save_adj(adj_rec.tocsr(), save_dir)
    utils.save_features(features_attacked, save_dir)

