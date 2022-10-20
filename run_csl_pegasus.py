# -*- coding: utf-8 -*-
import argparse
import torch
import random
from bert.utils.constants import CLS_ID, PAD_ID, SEP_ID, END_TOKEN, UNK_ID
import torch.multiprocessing as mp
from bert.transformer_builder import build_model
from bert.model_loader import load_model
from bert.utils.config import load_hyperparam
from bert.utils.seed import set_seed
from bert.utils.vocab import Vocab
from bert.utils.optimizers import AdamW, ConstantLRSchedule, Adafactor, TransformerSchedule, InvSquareRoot
from torch.nn.parallel import DistributedDataParallel
import torch.distributed as dist
from bert.model_saver import save_model
from bert.utils.tokenizer import BertTokenizer
from multiprocessing import Process,Manager, Lock
import time
from multiprocessing import Pool
from rouge import Rouge
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
rouge = Rouge()
smooth = SmoothingFunction().method1
def is_english_char(single_c):
    ord_c = ord(single_c)
    if (ord_c>=65 and ord_c<=90) or (ord_c>=97 and ord_c<=122) or ord_c==32:
        return True
    else:
        return False
def get_english_words(article):
    english_words = []
    index = 0
    word = ""
    while index<len(article):
        if is_english_char(article[index]):
            word += article[index]
            index+=1
            while index<len(article) and is_english_char(article[index]):
                if article[index]==" ":
                    english_words.append(word.strip())
                word += article[index]
                index+=1
            english_words.append(word.strip())
            word = ""
        index+=1
    return english_words

def get_proj_words(article):
    english_words = get_english_words(article)
    proj_dict = {}
    for word in english_words:
        gen_word = word.replace(" ", "").lower()
        if gen_word!=word and gen_word:
            proj_dict[gen_word] = word
    return proj_dict
def recover_gen_english_words(proj_dict, gen_article):
    words = get_english_words(gen_article)
    for word in words:
        if proj_dict.get(word) is not None:
            gen_article = gen_article.replace(word, proj_dict[word])
    return gen_article
def read_dataset(args, path):

    with open(path, mode="r", encoding="utf-8") as f:
        data = f.read().split("\n")
        print(len(data))
        #data = f.read().splitlines()
        data = [x.split("\t") for x in data]

    dataset = []
    for i in range(len(data)):
        if i%10000==0:
            print ("Dealed {}, Total {}".format(i, len(data)))
        if len(data[i])<2:
            continue
        if not "".join(data[i][1:]).strip() or not data[i][0].strip():
            continue
        article = data[i][1:]
        summ = data[i][0]
        src = []
        unk_sets = {}
        unk_num = len(args.vocab) - 1
        for sentence in article:
            all_words = args.tokenizer.tokenize(sentence)
            for t in all_words:
                t_id = args.vocab.get(t)
                #if t_id==UNK_ID and t!="”" and t!="“":
                if t_id==UNK_ID:
                    if unk_sets.get(t) is not None:
                        t_id = unk_sets[t]
                    else:
                        unk_sets[t] = unk_num
                        t_id = unk_num
                        unk_num -= 1
                src.append(t_id)
            src.append(SEP_ID)
        src = src[0:-1]
        tgt = []
        all_words = args.tokenizer.tokenize(summ)
        for t in all_words:
            t_id = args.vocab.get(t)
            if t_id == UNK_ID:
                if unk_sets.get(t) is not None:
                    t_id = unk_sets[t]
            tgt.append(t_id)
        tgt.append(args.vocab.get(END_TOKEN))
        if len(src)==0:
            continue
        src = src[0:args.seq_length] + [PAD_ID] * (args.seq_length - len(src))
        tgt = tgt[0:args.tgt_length] + [PAD_ID] * (args.tgt_length - len(tgt))
        dataset.append([src, tgt, unk_sets])
    return dataset

def batch_loader(args, trainset, proc_id, proc_num, test=False):
    procset = trainset[proc_id::proc_num]
    src = torch.LongTensor([example[0] for example in procset])
    tgt = torch.LongTensor([example[1] for example in procset])
    unk_sets = [example[2] for example in procset]
    instances_num = src.size()[0]
    if test:
        batch_size = args.test_batch_size
    else:
        batch_size = args.batch_size
    i = 0
    while True:
        if i*batch_size>=instances_num and not test:
            i = 0
        if i*batch_size>=instances_num and test:
            break
        src_batch = src[i * batch_size: (i + 1) * batch_size, :]
        tgt_batch = tgt[i * batch_size: (i + 1) * batch_size]
        unk_sets_batch = unk_sets[i * batch_size: (i + 1) * batch_size]
        i += 1
        yield src_batch, tgt_batch, unk_sets_batch



def build_optimizer(args, model):
    param_optimizer = list(model.named_parameters())
    '''tmp = []
    for n, p in param_optimizer:
        if "st" in n:
            tmp.append([n, p])
    param_optimizer = tmp'''

    no_decay = ['bias', 'gamma', 'beta']
    optimizer_grouped_parameters = [
                {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.01},
                {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.0}
    ]
    optimizer = Adafactor(optimizer_grouped_parameters, lr=args.learning_rate, relative_step=False)
    scheduler = InvSquareRoot(optimizer, warmup_steps=4000)
    return optimizer, scheduler

def train_model(args, gpu_id, rank, loader, model, optimizer, scheduler, steps):
    model.train()
    total_loss = 0
    total_steps = args.total_steps
    while True:

        if steps == total_steps + 1:
            break
        src, tgt, _ = next(loader)
        if gpu_id is not None:
            src = src.cuda(gpu_id)
            tgt = tgt.cuda(gpu_id)
        loss, _, _ = model(src, tgt)
        total_loss += loss.item()
        loss = loss / args.accumulation_steps
        if args.fp16:
            with args.amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()

        steps += 1
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        if steps % args.accumulation_steps == 0:
            optimizer.step()
            scheduler.step()
            model.zero_grad()
        #logging.info(loss)
        if steps % args.report_steps == 0 and \
                (not args.dist_train or (args.dist_train and rank == 0)):
            loss = total_loss / args.report_steps
            print ("Step {}, Loss is {}, lr is {}".format(steps, loss, optimizer.state_dict()['param_groups'][0]['lr']), flush=True)
            total_loss = 0
        if steps % args.save_steps==0 and \
                (not args.dist_train or (args.dist_train and rank == 0)):
            if args.dist_train:
                result, sentence_results = evaluate(args, model.module, gpu_id, read_dataset(args, args.dev_path), args.dev_path)
            else:
                result, sentence_results = evaluate(args, model, gpu_id, read_dataset(args, args.dev_path), args.dev_path)
            with open(args.result_path + "-{}".format(steps), "w", encoding="utf-8") as f:
                f.write("\n".join(sentence_results))
            save_model(model, args.output_model_path + "-{}".format(steps))
        model.train()





def cal_rouge_blue(args, candidate_datas, data_path):
    gold_datas = open(data_path, encoding="utf-8").read().split("\n")
    gold_datas = [" ".join(x.split("\t")[0]) for x in gold_datas if len(x.split("\t"))>=2]
    candidate_datas = [" ".join(x.replace("[unused2]", "").replace("##", "")) for x in candidate_datas]
    total = 0
    rouge_1, rouge_2, rouge_l, bleu = 0, 0, 0, 0
    print (len(gold_datas), len(candidate_datas))
    correct = 0
    for i in range(len(gold_datas)):
        scores = rouge.get_scores(hyps=candidate_datas[i], refs=gold_datas[i])
        rouge_1 += scores[0]['rouge-1']['f']
        rouge_2 += scores[0]['rouge-2']['f']
        rouge_l += scores[0]['rouge-l']['f']
        bleu += sentence_bleu(references=[gold_datas[i].split(" ")],
                              hypothesis=candidate_datas[i].split(" "),
                              smoothing_function=smooth)
        if gold_datas[i] == candidate_datas[i]:
            correct +=1
        total += 1
    rouge_1 /= total
    rouge_2 /= total
    rouge_l /= total
    bleu /= total
    acc = correct/total
    print("Rouge-1 is {}, Rouge-2 is {}, Rouge-L is {}, Bleu is {}, total is {}".format(rouge_1, rouge_2, rouge_l, bleu,
                                                                                        total), flush=True)
    print ("Acc is {}".format(acc), flush=True)
    return bleu



def evaluate(args,model, gpu_id, dataset, data_path):

    model.eval()
    results = []
    datas = open(data_path).read().splitlines()
    datas = [x.split("\t", 1)[1] for x in datas if len(x.split("\t")) >= 2]
    t1 = time.time()
    for index, (src_batch, tgt_batch, unk_sets) in enumerate(batch_loader(args, dataset, 0, 1, test=True)):
        if index%100==0:
            print ("Dealed {}, Total {}".format(index*args.batch_size, len(dataset)), flush=True)
        src_batch = src_batch.to(gpu_id)
        tgt_batch = tgt_batch.to(gpu_id)
        with torch.no_grad():
            pred_tokens = model.inference_forward_beam_search(src_batch)
            #pred_tokens = model.inference_forward_greedy(src_batch)
            #pred_tokens = model.inference_forward_beam_search_without_tgt(src_batch, tgt_batch)
            #pred_tokens = model.greedy_sample(src_batch)
        for i in range(len(pred_tokens)):
            x = "".join([args.vocab.i2w[x] for x in pred_tokens[i][0]])
            x = x.split("[unused2]")[0]
            x = x.replace("[UNK]", "").replace("[unused2]", "").replace("##", "")
            for unk_word in unk_sets[i]:
                x = x.replace(args.vocab.i2w[unk_sets[i][unk_word]].replace("##", ""), unk_word)
            if x=="":
                x = "111"
            porj_dict = get_proj_words(datas[index*args.test_batch_size + i])
            x = recover_gen_english_words(porj_dict, x)
            results.append(x)
    t2 = time.time()
    print ("Total time:", t2-t1, flush=True)


    bleu = cal_rouge_blue(args, results, data_path)
    return bleu, results


def worker(proc_id, gpu_ranks, args, model, trainset):
    """
    Args:
        proc_id: The id of GPU for single GPU mode;
                 The id of process (and GPU) for multiprocessing distributed mode.
        gpu_ranks: List of ranks of each process.
    """
    set_seed(args.seed)
    if args.dist_train:
        rank = gpu_ranks[proc_id]
        gpu_id = gpu_ranks[proc_id]
    elif args.single_gpu:
        rank = None
        gpu_id = proc_id
    else:
        rank = None
        gpu_id = None

    if gpu_id is not None:
        torch.cuda.set_device(gpu_id)
        model.cuda(gpu_id)

    steps = 0
    optimizer, scheduler = build_optimizer(args, model)

    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)
        steps = 0
        args.amp = amp

    if args.dist_train:
        # Initialize multiprocessing distributed training environment.
        dist.init_process_group(backend=args.backend,
                                init_method=args.master_ip,
                                world_size=args.world_size,
                                rank=rank)
        print('inited distri group rank %d' % rank, flush=True)
        model = DistributedDataParallel(model, device_ids=[gpu_id], find_unused_parameters=True)
        print("Worker %d is training ... " % rank, flush=True)
    else:
        print("Worker is training ...")
    if gpu_ranks is not None:
        train_loader = batch_loader(args, trainset, proc_id, len(gpu_ranks))
    else:
        train_loader = batch_loader(args, trainset, proc_id, 1)
    train_model(args, gpu_id, rank, train_loader, model, optimizer, scheduler, steps)



def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Path options.
    parser.add_argument("--pretrained_model_path", default=None, type=str,
                        help="Path of the pretrained model.")
    parser.add_argument("--output_model_path", default="./models/classifier_model.bin", type=str,
                        help="Path of the output model.")
    parser.add_argument("--vocab_path", type=str,
                        help="Path of the vocabulary file.")
    parser.add_argument("--train_path", type=str,
                        help="Path of the trainset.")
    parser.add_argument("--dev_path", type=str,
                        help="Path of the devset.")
    parser.add_argument("--dev_label_path", type=str,help="Path of the devset.")
    parser.add_argument("--test_path", type=str,
                        help="Path of the testset.")
    parser.add_argument("--config_path", default="./models/bert_base_config.json", type=str,
                        help="Path of the config file.")
    parser.add_argument("--result_path", type=str, help="Path of result path")

    # Model options.
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size.")
    parser.add_argument("--test_batch_size", type=int, default=128,
                        help="Batch size.")
    parser.add_argument("--seq_length", type=int, default=128,
                        help="Sequence length.")
    parser.add_argument("--max_length", type=int, default=512,
                        help="Max length for position embedding.")
    parser.add_argument("--tgt_length", type=int, default=32, help="target sequence length")
    parser.add_argument("--min_tgt_length", type=int, default=1, help="target sequence length")
    parser.add_argument("--beam_size", type=int, default=1, help="target sequence length")
    parser.add_argument("--bs_group", type=int, default=2, help="group size of diverse beam search")
    parser.add_argument("--factorized_embedding_parameterization", action="store_true",
                        help="Factorized embedding parameterization.")
    parser.add_argument("--parameter_sharing", action="store_true", help="Parameter sharing.")
    parser.add_argument("--remove_transformer_bias", action="store_true", help="remove linear bias in transformer")
    parser.add_argument("--feed_forward", type=str, default="normal", choices=["normal", "gated"], help="feed forward type")
    parser.add_argument("--layernorm_position", type=str, default="post", choices=["post", "pre"], help="layer norm position")
    parser.add_argument("--hidden_act", type=str, default="relu", help="layer norm position")
    parser.add_argument("--gpu_ranks", default=[], nargs='+', type=int, help="List of ranks of each process.")
    parser.add_argument("--accumulation_steps", default=1, type=int)
    parser.add_argument("--report_steps", default=100, type=int)
    parser.add_argument("--save_steps", default=1000, type=int)
    parser.add_argument("--total_steps", default=50000, type=int)

    # Tokenizer options.
    parser.add_argument("--tokenizer", choices=["bert", "char", "space"], default="bert",
                        help="Specify the tokenizer."
                             "Original Google BERT uses bert tokenizer on Chinese corpus."
                             "Char tokenizer segments sentences into characters."
                             "Space tokenizer segments sentences into words according to space."
                        )

    # Optimizer options.
    parser.add_argument("--soft_targets", action='store_true',
                        help="Train model with logits.")
    parser.add_argument("--learning_rate", type=float, default=2e-5,
                        help="Learning rate.")
    parser.add_argument("--warmup", type=float, default=0.1,
                        help="Warm up value.")
    parser.add_argument("--fp16", action='store_true',
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument("--fp16_opt_level", choices=["O0", "O1", "O2", "O3"], default='O0',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")

    # Training options.
    parser.add_argument("--dropout", type=float, default=0.5,
                        help="Dropout.")
    parser.add_argument("--epochs_num", type=int, default=3,
                        help="Number of epochs.")
    parser.add_argument("--seed", type=int, default=7,
                        help="Random seed.")
    parser.add_argument("--target", type=str, default="mlm_unilm", help="target of model")
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--dist_train", action="store_true")
    parser.add_argument("--single_gpu", action="store_true")
    parser.add_argument("--backend", choices=["nccl", "gloo"], default="nccl", type=str, help="Distributed backend.")
    parser.add_argument("--master_ip", default="tcp://localhost:12345", type=str,
                        help="IP-Port of master for training.")


    args = parser.parse_args()

    # Load the hyperparameters from the config file.
    args = load_hyperparam(args)
    print (args.output_model_path, flush=True)
    set_seed(args.seed)
    # Load vocabulary.
    vocab = Vocab()
    vocab.load(args.vocab_path)
    args.vocab = vocab
    args.tokenizer = globals()[args.tokenizer.capitalize() + "Tokenizer"](args)
    if args.single_gpu:
        args.gpu_id = args.gpu_ranks[0]
    # Build classification model.
    model = build_model(args)

    # Load or initialize parameters.


    # Build tokenizer.

    if args.test:
        print("Test set evaluation.")
        model = load_model(model, args.output_model_path)
        #model.load_state_dict(torch.load(args.output_model_path, map_location=torch.device('cpu')))
        model = model.cuda(args.gpu_ranks[0])
        result, sentence_results = evaluate(args, model,args.gpu_ranks[0], read_dataset(args, args.dev_path), args.dev_path)
        gold_datas = open(args.dev_path).read().split("\n")
        results = []
        for i in range(len(sentence_results)):
            results.append("输入: {}".format("\t".join(gold_datas[i].split("\t")[1:])))
            results.append("预测: {}".format(sentence_results[i]))
        results = sentence_results
        with open("./article_comment.result", "w", encoding="utf-8") as f:
            f.write("\n".join(results))
    else:
        if args.pretrained_model_path is not None:
            model = load_model(model, args.pretrained_model_path)
        # Training phase.
        trainset = read_dataset(args, args.train_path)
        random.shuffle(trainset)



        args.ranks_num = len(args.gpu_ranks)
        args.world_size = len(args.gpu_ranks)
        print (args.ranks_num, args.world_size, args.gpu_ranks)
        if args.dist_train:
            # Multiprocessing distributed mode.
            mp.spawn(worker, nprocs=args.ranks_num, args=(args.gpu_ranks, args, model, trainset), daemon=False)
        elif args.single_gpu:
            # Single GPU mode.
            worker(args.gpu_id, None, args, model, trainset)
        else:
            # CPU mode.
            worker(None, None, args, model, trainset)


if __name__ == "__main__":
    main()



