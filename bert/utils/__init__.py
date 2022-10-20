from bert.utils.act_fun import gelu, gelu_fast, relu

str2act = {"gelu": gelu, "gelu_fast":gelu_fast, "relu":relu}

__all__ = ["str2act", "gelu", "gelu_fast", "relu"]