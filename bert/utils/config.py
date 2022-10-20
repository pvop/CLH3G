# -*- encoding:utf-8 -*-
import json


def load_hyperparam(args):
    with open(args.config_path, mode="r", encoding="utf-8") as f:
        param = eval(f.read())
    args.emb_size = param.get("emb_size", 768)
    args.hidden_size = param.get("hidden_size", 768)
    args.feedforward_size = param.get("feedforward_size", 3072)
    args.heads_num = param.get("heads_num", 12)
    args.layers_num = param.get("layers_num", 12)
    args.dropout = param.get("dropout", 0.1)
    for k, v in param.items():
        args.__dict__[k] = v

    return args
