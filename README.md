# TitleStylist
Source code for our "CLH3G" paper at EMNLP 2022: Contrastive Learning enhanced Author-Style Headline Generation.

## Requirements
### Python packages
- Pytorch
- rouge
- nltk

In order to install them, you can run this command:

```
pip install -r requirements.txt
```

### Dataset
You can download original dataset, our processed dataset and trained CLH3G model from https://drive.google.com/file/d/1vHDhhYmSEb4EmIEshMT2NnwggiXYtsrV/view?usp=sharing


## Usage
1. Download bert-base-chinese from huggingface https://huggingface.co/bert-base-chinese/tree/main, and convert bert model to this project (which is in google driver already) as:
```
python convert_bert_from_huggingface_to_bertpytorch.py
```
2. You can train CLH3G model with 4 GPUs (total batch size 96, 24 for each GPU) as:
```
python run_clh3g.py --config_path configs/clh3g_train.json --gpu_ranks 0 1 2 3
```
note: You can use accumulation_steps to achieve max batch sizes on GPU with limit memory.

3. You can eval CLH3G model with 1 GPU as:
```
python run_clh3g.py --config_path configs/clh3g_train.json --gpu_ranks 0
```
4. You can train and eval the headline style of generated headlines as:
```
python run_contra.py --config_path configs/contrajson --gpu_ranks 0
```

