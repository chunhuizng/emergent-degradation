import os
import torch
import grb.utils as utils
from grb.dataset import Dataset
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

from numba import cuda
cuda.select_device(0)
cuda.close()
cuda.select_device(0)

dataset_name = "grb-cora"
dataset = Dataset(name=dataset_name,
                  data_dir="../data/",
                  mode="full",
                  feat_norm="arctan")

from gatdp import GAT
model_name = "gatdp0.1"
model = GAT(in_features=dataset.num_features,
            out_features=dataset.num_classes,
            hidden_features=64,
            n_layers=3,
            n_heads=6,
            layer_norm=False,
            dropout=0.5)


print("Number of parameters: {}.".format(utils.get_num_params(model)))
print(model)

save_dir = "./saved_models/{}/{}".format(dataset_name, model_name)
save_name = "model.pt"
device = "cuda:0"
feat_norm = None
train_mode = "inductive"  # "transductive"
#from grb.trainer.trainer import Trainer
from trainer import Trainer
trainer = Trainer(dataset=dataset,
                  optimizer=torch.optim.Adam(model.parameters(), lr=0.01),
                  loss=torch.nn.functional.cross_entropy,
                  lr_scheduler=False,
                  early_stop=True,
                  early_stop_patience=500,
                  feat_norm=feat_norm,
                  device=device)
trainer.train(model=model,
              n_epoch=1000,
              eval_every=1,
              save_after=0,
              save_dir=save_dir,
              save_name=save_name,
              train_mode=train_mode,
              verbose=False,
              )

model = torch.load(os.path.join(save_dir, save_name))
model = model.to(device)
model.eval()

# by trainer
pred = trainer.inference(model)
# by utils
pred = utils.inference(model,
                       features=dataset.features,
                       feat_norm=feat_norm,
                       adj=dataset.adj,
                       adj_norm_func=model.adj_norm_func,
                       device=device)
# by trainer
test_score = trainer.evaluate(model, dataset.test_mask)
print("Test score: {:.4f}".format(test_score))

# by utils
test_score = utils.evaluate(model,
                            features=dataset.features,
                            adj=dataset.adj,
                            labels=dataset.labels,
                            feat_norm=feat_norm,
                            adj_norm_func=model.adj_norm_func,
                            mask=dataset.test_mask,
                            device=device)
print("Test score: {:.4f}".format(test_score))


