import torch
import argparse
import collections

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--input_model_path", type=str, default="bert_models/pytorch_model.bin",
                        help=".")
parser.add_argument("--output_model_path", type=str, default="bert_models/bert_base.bin",
                        help=".")
parser.add_argument("--layers_num", type=int, default=12, help=".")

args = parser.parse_args()
path = args.input_model_path

input_model = torch.load(args.input_model_path, map_location='cpu')
output_model = collections.OrderedDict()

output_model["src_embedding.word_embedding.weight"] = input_model["bert.embeddings.word_embeddings.weight"]
output_model["src_embedding.abs_position_embedding.weight"] = input_model["bert.embeddings.position_embeddings.weight"]
output_model["src_embedding.segment_embedding.weight"] = input_model["bert.embeddings.token_type_embeddings.weight"]
output_model["src_embedding.layer_norm.gamma"] = input_model["bert.embeddings.LayerNorm.gamma"]
output_model["src_embedding.layer_norm.beta"] = input_model["bert.embeddings.LayerNorm.beta"]

for i in range(args.layers_num):
    output_model["encoder.transformer." + str(i) + ".self_attn.linear_layers.0.weight"] = input_model["bert.encoder.layer." + str(i) + ".attention.self.query.weight"]
    output_model["encoder.transformer." + str(i) + ".self_attn.linear_layers.0.bias"] = input_model["bert.encoder.layer." + str(i) + ".attention.self.query.bias"]
    output_model["encoder.transformer." + str(i) + ".self_attn.linear_layers.1.weight"] = input_model["bert.encoder.layer." + str(i) + ".attention.self.key.weight"]
    output_model["encoder.transformer." + str(i) + ".self_attn.linear_layers.1.bias"] = input_model["bert.encoder.layer." + str(i) + ".attention.self.key.bias"]
    output_model["encoder.transformer." + str(i) + ".self_attn.linear_layers.2.weight"] = input_model["bert.encoder.layer." + str(i) + ".attention.self.value.weight"]
    output_model["encoder.transformer." + str(i) + ".self_attn.linear_layers.2.bias"] = input_model["bert.encoder.layer." + str(i) + ".attention.self.value.bias"]
    output_model["encoder.transformer." + str(i) + ".self_attn.final_linear.weight"] = input_model["bert.encoder.layer." + str(i) + ".attention.output.dense.weight"]
    output_model["encoder.transformer." + str(i) + ".self_attn.final_linear.bias"] = input_model["bert.encoder.layer." + str(i) + ".attention.output.dense.bias"]
    output_model["encoder.transformer." + str(i) + ".layer_norm_1.gamma"] = input_model["bert.encoder.layer." + str(i) + ".attention.output.LayerNorm.gamma"]
    output_model["encoder.transformer." + str(i) + ".layer_norm_1.beta"] = input_model["bert.encoder.layer." + str(i) + ".attention.output.LayerNorm.beta"]
    output_model["encoder.transformer." + str(i) + ".feed_forward.linear_1.weight"] = input_model["bert.encoder.layer." + str(i) + ".intermediate.dense.weight"]
    output_model["encoder.transformer." + str(i) + ".feed_forward.linear_1.bias"] = input_model["bert.encoder.layer." + str(i) + ".intermediate.dense.bias"]
    output_model["encoder.transformer." + str(i) + ".feed_forward.linear_2.weight"] = input_model["bert.encoder.layer." + str(i) + ".output.dense.weight"]
    output_model["encoder.transformer." + str(i) + ".feed_forward.linear_2.bias"] = input_model["bert.encoder.layer." + str(i) + ".output.dense.bias"]
    output_model["encoder.transformer." + str(i) + ".layer_norm_2.gamma"] = input_model["bert.encoder.layer." + str(i) + ".output.LayerNorm.gamma"]
    output_model["encoder.transformer." + str(i) + ".layer_norm_2.beta"] = input_model["bert.encoder.layer." + str(i) + ".output.LayerNorm.beta"]




torch.save(output_model, args.output_model_path)
