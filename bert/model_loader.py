# -*- encoding:utf-8 -*-
import torch
import collections


def load_model(model, model_path):
    model_state_dict = torch.load(model_path, map_location=torch.device('cpu'))
    new_state_dict = collections.OrderedDict()
    if "model_state_dict" in model_state_dict:
        model_state_dict = model_state_dict["model_state_dict"]
    for k, v in model_state_dict.items():
        if k.startswith("module"):
            name = k[7:]
        else:
            name = k
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict, strict=False)

    return model
