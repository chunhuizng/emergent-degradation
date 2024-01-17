"""Configuration for reproducing leaderboard of grb-citeseer dataset."""
import torch
import torch.nn.functional as F

from grb.evaluator import metric

standard_train_model_list = ["gcn",
                             "gcn_ln",
                             "robustgcn",
                             "gin",
                             "gat",
                             "gatdp",
                             "gatdp_moe"]

adv_train_model_list = ["gat_at",
                        "gatdp_at",
                        "gatdp_moe_at",
                        "robustgcn_at",
                        "gin_at"]

model_list = ["gcn",
              "gcn_ln",
              "robustgcn",
              "gin",
              "gat",
              "gatdp",
              "gatdp_moe",
              "gat_at",
              "gatdp_at",
              "gatdp_moe_at",
              "robustgcn_at",
              "gin_at"]

model_list_basic = ["gcn",
                    "graphsage",
                    "sgcn",
                    "tagcn",
                    "appnp",
                    "gin",
                    "gat"]

modification_attack_list = ["dice",
                            "rand",
                            "flip",
                            "fga",
                            "nea",
                            "pgd",
                            "prbcd",
                            "stack"]

injection_attack_list = ["rand",
                         "fgsm",
                         "pgd",
                         "speit",
                         "tdgia"]

model_sur_list = ["gcn"]


def build_model(model_name, num_features, num_classes):
    """Hyper-parameters are determined by auto training, refer to grb.utils.trainer.AutoTrainer."""
    if model_name in ["gatdp_moe", "gatdp_moe_at"]:
        from gatdp_moe import GATDP_moe
        model = GATDP_moe(in_features=num_features,
                          out_features=num_classes,
                          hidden_features=64,
                          n_layers=3,
                          n_heads=6,
                          layer_norm=True if "ln" in model_name else False,
                          dropout=0.6)
        # num_experts = 4, k = 1, i + 1, base_attack = 150
        train_params = {
            "lr": 0.005,
            "n_epoch": 5000,
            "early_stop": True,
            "early_stop_patience": 500,
            "train_mode": "inductive",
        }
        return model, train_params
    if model_name in ["gatdp", "gatdp_at"]:
        from gatdp import GATDP
        model = GATDP(in_features=num_features,
                      out_features=num_classes,
                      hidden_features=64,
                      n_layers=3,
                      n_heads=6,
                      layer_norm=True if "ln" in model_name else False,
                      dropout=0.6,
                      dp_rate=0.3)

        train_params = {
            "lr": 0.005,
            "n_epoch": 5000,
            "early_stop": True,
            "early_stop_patience": 500,
            "train_mode": "inductive",
        }
        return model, train_params
    if model_name in ["gcn", "gcn_ln", "gcn_at", "gcn_ln_at"]:
        from grb.model.torch import GCN
        model = GCN(in_features=num_features,
                    out_features=num_classes,
                    hidden_features=128,
                    n_layers=3,
                    layer_norm=True if "ln" in model_name else False,
                    dropout=0.7)
        train_params = {
            "lr": 0.001,
            "n_epoch": 5000,
            "early_stop": True,
            "early_stop_patience": 500,
            "train_mode": "inductive",
        }
        return model, train_params
    if model_name in ["graphsage", "graphsage_ln", "graphsage_at", "graphsage_ln_at"]:
        from grb.model.torch import GraphSAGE
        model = GraphSAGE(in_features=num_features,
                          out_features=num_classes,
                          hidden_features=256,
                          n_layers=5,
                          layer_norm=True if "ln" in model_name else False,
                          dropout=0.5)
        train_params = {
            "lr": 0.0001,
            "n_epoch": 5000,
            "early_stop": True,
            "early_stop_patience": 500,
            "train_mode": "inductive",
        }
        return model, train_params
    if model_name in ["sgcn", "sgcn_ln", "sgcn_at", "sgcn_ln_at"]:
        from grb.model.torch import SGCN
        model = SGCN(in_features=num_features,
                     out_features=num_classes,
                     hidden_features=256,
                     n_layers=4,
                     k=4,
                     layer_norm=True if "ln" in model_name else False,
                     dropout=0.5)
        train_params = {
            "lr": 0.01,
            "n_epoch": 5000,
            "early_stop": True,
            "early_stop_patience": 500,
            "train_mode": "inductive",
        }
        return model, train_params
    if model_name in ["tagcn", "tagcn_ln", "tagcn_at", "tagcn_ln_at"]:
        from grb.model.torch import TAGCN
        model = TAGCN(in_features=num_features,
                      out_features=num_classes,
                      hidden_features=256,
                      n_layers=3,
                      k=2,
                      layer_norm=True if "ln" in model_name else False,
                      dropout=0.5)
        train_params = {
            "lr": 0.005,
            "n_epoch": 5000,
            "early_stop": True,
            "early_stop_patience": 500,
            "train_mode": "inductive",
        }
        return model, train_params
    if model_name in ["appnp", "appnp_ln", "appnp_at", "appnp_ln_at"]:
        from grb.model.torch import APPNP
        model = APPNP(in_features=num_features,
                      out_features=num_classes,
                      hidden_features=128,
                      n_layers=3,
                      k=3,
                      layer_norm=True if "ln" in model_name else False,
                      dropout=0.5)
        train_params = {
            "lr": 0.001,
            "n_epoch": 5000,
            "early_stop": True,
            "early_stop_patience": 500,
            "train_mode": "inductive",
        }
        return model, train_params
    if model_name in ["gin", "gin_ln", "gin_at", "gin_ln_at"]:
        from grb.model.torch import GIN
        model = GIN(in_features=num_features,
                    out_features=num_classes,
                    hidden_features=256,
                    n_layers=2,
                    layer_norm=True if "ln" in model_name else False,
                    dropout=0.6)
        train_params = {
            "lr": 0.0001,
            "n_epoch": 5000,
            "early_stop": True,
            "early_stop_patience": 500,
            "train_mode": "inductive",
        }
        return model, train_params
    if model_name in ["gat", "gat_ln", "gat_at", "gat_ln_at"]:
        from grb.model.dgl import GAT
        model = GAT(in_features=num_features,
                    out_features=num_classes,
                    hidden_features=64,
                    n_layers=3,
                    n_heads=6,
                    layer_norm=True if "ln" in model_name else False,
                    dropout=0.6)
        train_params = {
            "lr": 0.005,
            "n_epoch": 5000,
            "early_stop": True,
            "early_stop_patience": 500,
            "train_mode": "inductive",
        }
        return model, train_params
    if model_name in ["robustgcn", "robustgcn_at"]:
        from grb.defense import RobustGCN
        model = RobustGCN(in_features=num_features,
                          out_features=num_classes,
                          hidden_features=64,
                          n_layers=2,
                          dropout=0.5)
        train_params = {
            "lr": 0.001,
            "n_epoch": 5000,
            "early_stop": True,
            "early_stop_patience": 500,
            "train_mode": "inductive",
        }
        return model, train_params
    if model_name in ["gcnsvd", "gcnsvd_ln"]:
        from grb.defense.gcnsvd import GCNSVD

        model = GCNSVD(in_features=num_features,
                       out_features=num_classes,
                       hidden_features=128,
                       n_layers=3,
                       dropout=0.5)
        train_params = {
            "lr": 0.001,
            "n_epoch": 5000,
            "early_stop": True,
            "early_stop_patience": 500,
            "train_mode": "inductive",
        }
        return model, train_params
    if model_name in ["gcnguard"]:
        from grb.defense import GCNGuard

        model = GCNGuard(in_features=num_features,
                         out_features=num_classes,
                         hidden_features=128,
                         n_layers=3,
                         dropout=0.5)
        train_params = {
            "lr": 0.001,
            "n_epoch": 5000,
            "early_stop": True,
            "early_stop_patience": 500,
            "train_mode": "inductive",
        }
        return model, train_params
    if model_name in ["gatguard"]:
        from grb.defense import GATGuard

        model = GATGuard(in_features=num_features,
                         out_features=num_classes,
                         hidden_features=64,
                         n_heads=6,
                         n_layers=3,
                         dropout=0.5)
        train_params = {
            "lr": 0.001,
            "n_epoch": 5000,
            "early_stop": True,
            "early_stop_patience": 500,
            "train_mode": "inductive",
        }
        return model, train_params


def build_optimizer(model, lr):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    return optimizer


def build_loss():
    return F.nll_loss


def build_metric():
    return metric.eval_acc


def build_attack(attack_name, device="cpu", args=None, mode="modification"):
    if mode == "modification":
        if attack_name == "dice":
            from grb.attack.modification import DICE

            attack = DICE(n_edge_mod=args.n_edge_mod,
                          ratio_delete=0.6,
                          device=device)
            return attack
        if attack_name == "fga":
            from grb.attack.modification import FGA

            attack = FGA(n_edge_mod=args.n_edge_mod,
                         device=device)
            return attack
        if attack_name == "flip":
            from grb.attack.modification import FLIP

            attack = FLIP(n_edge_mod=args.n_edge_mod,
                          flip_type=args.flip_type,
                          mode="descend",
                          device=device)
            return attack
        if attack_name == "rand":
            from grb.attack.modification import RAND

            attack = RAND(n_edge_mod=args.n_edge_mod,
                          device=device)
            return attack
        if attack_name == "nea":
            from grb.attack.modification import NEA

            attack = NEA(n_edge_mod=args.n_edge_mod,
                         device=device)
            return attack
        if attack_name == "stack":
            from grb.attack.modification import STACK

            attack = STACK(n_edge_mod=args.n_edge_mod,
                           device=device)
            return attack
        if attack_name == "pgd":
            from grb.attack.modification import PGD

            attack = PGD(epsilon=args.epsilon,
                         n_epoch=args.attack_epoch,
                         n_node_mod=args.n_node_mod,
                         n_edge_mod=args.n_edge_mod,
                         feat_lim_min=args.feat_lim_min,
                         feat_lim_max=args.feat_lim_max,
                         early_stop=args.early_stop,
                         device=device)
            return attack
        if attack_name == "prbcd":
            from grb.attack.modification import PRBCD

            attack = PRBCD(epsilon=args.epsilon,
                           n_epoch=args.attack_epoch,
                           n_node_mod=args.n_node_mod,
                           n_edge_mod=args.n_edge_mod,
                           feat_lim_min=args.feat_lim_min,
                           feat_lim_max=args.feat_lim_max,
                           early_stop=args.early_stop,
                           device=device)
            return attack
    elif mode == "injection":
        if attack_name == "rand":
            from grb.attack.injection import RAND

            attack = RAND(n_inject_max=args.n_inject_max,
                          n_edge_max=args.n_edge_max,
                          feat_lim_min=args.feat_lim_min,
                          feat_lim_max=args.feat_lim_max,
                          device=device)
            return attack
        elif attack_name == "fgsm":
            from grb.attack.injection import FGSM

            attack = FGSM(epsilon=args.attack_lr,
                          n_epoch=args.attack_epoch,
                          n_inject_max=args.n_inject_max,
                          n_edge_max=args.n_edge_max,
                          feat_lim_min=args.feat_lim_min,
                          feat_lim_max=args.feat_lim_max,
                          early_stop=args.early_stop,
                          device=device)
            return attack
        elif attack_name == "pgd":
            from grb.attack.injection import PGD

            attack = PGD(epsilon=args.attack_lr,
                         n_epoch=args.attack_epoch,
                         n_inject_max=args.n_inject_max,
                         n_edge_max=args.n_edge_max,
                         feat_lim_min=args.feat_lim_min,
                         feat_lim_max=args.feat_lim_max,
                         early_stop=args.early_stop,
                         device=device)
            return attack
        elif attack_name == "speit":
            from grb.attack.injection import SPEIT

            attack = SPEIT(lr=args.attack_lr,
                           n_epoch=args.attack_epoch,
                           n_inject_max=args.n_inject_max,
                           n_edge_max=args.n_edge_max,
                           feat_lim_min=args.feat_lim_min,
                           feat_lim_max=args.feat_lim_max,
                           early_stop=args.early_stop,
                           device=device)
            return attack
        elif attack_name == "tdgia":
            from grb.attack.injection import TDGIA

            attack = TDGIA(lr=args.attack_lr,
                           n_epoch=args.attack_epoch,
                           n_inject_max=args.n_inject_max,
                           n_edge_max=args.n_edge_max,
                           feat_lim_min=args.feat_lim_min,
                           feat_lim_max=args.feat_lim_max,
                           early_stop=args.early_stop,
                           inject_mode='random',
                           sequential_step=1.0,
                           device=device)
            return attack
        elif attack_name == "tdgia_random":
            from grb.attack.injection.tdgia import TDGIA

            attack = TDGIA(lr=args.attack_lr,
                           n_epoch=args.attack_epoch,
                           n_inject_max=args.n_inject_max,
                           n_edge_max=args.n_edge_max,
                           feat_lim_min=args.feat_lim_min,
                           feat_lim_max=args.feat_lim_max,
                           early_stop=args.early_stop,
                           inject_mode='random',
                           device=device)
            return attack
        elif attack_name == "tdgia_uniform":
            from grb.attack.injection import TDGIA

            attack = TDGIA(lr=args.attack_lr,
                           n_epoch=args.attack_epoch,
                           n_inject_max=args.n_inject_max,
                           n_edge_max=args.n_edge_max,
                           feat_lim_min=args.feat_lim_min,
                           feat_lim_max=args.feat_lim_max,
                           early_stop=args.early_stop,
                           inject_mode='uniform',
                           sequential_step=1.0,
                           device=device)
            return attack
    else:
        raise NotImplementedError
