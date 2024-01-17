import os
import torch
import grb.utils as utils
from grb.dataset import Dataset
from grb.model.torch import GCN
from grb.model.dgl import GAT
from moedp_model.gatdp import GAT
from grb.defense import AdvTrainer
from grb.utils.normalize import GCNAdjNorm
from grb.attack.injection import FGSM


def main():
    from numba import cuda
    cuda.select_device(0)
    cuda.close()
    cuda.select_device(0)

    dataset_name = 'grb-flickr'
    dataset = Dataset(name=dataset_name,
                      data_dir="../data/",
                      mode='full',
                      feat_norm='arctan')
    adj = dataset.adj
    features = dataset.features
    labels = dataset.labels
    num_features = dataset.num_features
    num_classes = dataset.num_classes


    # # #GAT
    #from grb.model.dgl import GAT
    from gatdp import GAT
    # from gatdp_moe import GAT_moe
    model_name = "gatdp1"
    model = GAT(in_features=dataset.num_features,
                out_features=dataset.num_classes,
                hidden_features=64,
                n_layers=3,
                n_heads=6,
                adj_norm_func=None,
                layer_norm=False,
                dropout=0.5)



    print("Number of parameters: {}.".format(utils.get_num_params(model)))
    print(model)

    device = 'cuda:0'

    attack = FGSM(epsilon=0.01,
                  n_epoch=10,
                  n_inject_max=200,
                  n_edge_max=100,
                  feat_lim_min=features.min(),
                  feat_lim_max=features.max(),
                  early_stop=False,
                  device=device,
                  verbose=False)

    save_dir = "./saved_models/{}/{}_at".format(dataset_name, model_name)
    save_name = "model.pt"
    device = "cuda:0"
    feat_norm = None
    train_mode = "inductive"  # "transductive"

    trainer = AdvTrainer(dataset=dataset,
                         attack=attack,
                         optimizer=torch.optim.Adam(model.parameters(), lr=0.01),
                         loss=torch.nn.functional.cross_entropy,
                         lr_scheduler=False,
                         early_stop=True,
                         early_stop_patience=500,
                         device=device)
    trainer.train(model=model,
                  n_epoch=10000,
                  eval_every=1,
                  save_after=0,
                  save_dir=save_dir,
                  save_name=save_name,
                  train_mode=train_mode,
                  verbose=False)


    # by trainer
    test_score = trainer.evaluate(model, dataset.test_mask)
    print("Test score: {:.4f}".format(test_score))

    test_score = utils.evaluate(model,
                                features=dataset.features,
                                adj=dataset.adj,
                                labels=dataset.labels,
                                feat_norm=feat_norm,
                                adj_norm_func=model.adj_norm_func,
                                mask=dataset.test_mask,
                                device=device)
    print("Test score: {:.4f}".format(test_score))

if __name__ == '__main__':
    main()



