# -*- encoding:utf-8 -*-
import torch
import collections


def save_model(model, model_path):
    if hasattr(model, "module"):
        torch.save(model.module.state_dict(), model_path)
    else:
        torch.save(model.state_dict(), model_path)
        

def save_checkpoint(checkpoint, model_path):
    torch.save(checkpoint, model_path)
