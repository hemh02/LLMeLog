import argparse
import torch
import pytorch_lightning as pl
from src.dataset import HSFloader, ADloader
from src.models import HSFencoder, LLMeLog
from src.utils import get_abs_path
from src.dataset import load_json
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import pandas as pd
from transformers import AutoModel, AutoTokenizer
import json


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')



def gen_encoder_embedding(sent, tokenizer, model):
    if (len(sent) >= 500): sent = sent[:500]
    inputs = tokenizer(sent, return_tensors="pt")
    output_pooler = model(**inputs)[1][0]   
    return output_pooler.tolist()

def gen_embedding_dict(path):
    data = pd.read_csv(path + '/sent.csv')
    tokenizer = AutoTokenizer.from_pretrained("./new_encoder/")
    model = AutoModel.from_pretrained("./new_encoder")  
    row, _ = data.shape    
    
    res = {}
    res["0"] = gen_encoder_embedding('[PAD]', tokenizer, model)

    for i in tqdm(range(row)):
        sent_emd = np.array(gen_encoder_embedding(data.iloc[i, 1], tokenizer, model)) 
        emd = sent_emd
        res[str(i + 1)] = emd.tolist()

    emd_dict_json = json.dumps(res, ensure_ascii=False, indent=4)
    with open (path + '/emd_dict.json', 'w+', encoding='utf-8') as file:
        file.write(emd_dict_json)  


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hard_device", default='cuda', type=str, help="cpu or cuda")
    parser.add_argument("--gpu_index", default=0, type=int, help='gpu index, one of [0,1,2,3,...]')
    parser.add_argument("--load_checkpoint", nargs='?', const=True, default=False, type=str2bool,
                        help="one of [True, False]")
    parser.add_argument('--model_save_path', default='checkpoint', type=str)
    parser.add_argument('--epochs', default=10, type=int, help='epochs')
    parser.add_argument('--batch_size', default=4, type=int, help='batch_size, 4 for hsf, 256 for classifier')
    parser.add_argument('--warmup_epochs', default=8, type=int, help='warmup')
    parser.add_argument('--lr', default=2e-4, type=float, help='learning rate, 2e-4 for hsf, 3e-4 for classifier')
    parser.add_argument('--accumulate_grad_batches',
                        default=16,
                        type=int,
                        help='accumulate_grad_batches')
    parser.add_argument('--mode', default='train', type=str,
                        help='one of [train, test, preproc]')
    parser.add_argument('--encoder', default='None', type=str,
                        help='train encoder, str None for classification')
    parser.add_argument('--anomal_class', default=0, type=int,
                        help='LLM anomal_class num')
    parser.add_argument('--dataset', default='hdfs', type=str,
                        help='dataset name')
    
    arguments = parser.parse_args()
    if arguments.hard_device == 'cpu':
        arguments.device = torch.device(arguments.hard_device)
    else:
        arguments.device = torch.device(f'cuda:{arguments.gpu_index}')
    print(arguments)
    return arguments


def main():
    args = parse_args()
    if args.encoder != 'None':
        path = 'data/' + args.dataset + '/sent.csv'
        sent_data = pd.read_csv(path)
        row, _ = sent_data.shape
        for i in range(row): 
            if sent_data.iloc[i, 2] == 1: args.anomal_class += 1
        model = HSFencoder(args).to(args.hard_device)
        train_loader = HSFloader('data/' + args.dataset + '/train.json',
                                            batch_size=args.batch_size,
                                            shuffle=True,
                                            num_workers=4)
        valid_loader = HSFloader('data/' + args.dataset + '/dev.json',
                                            batch_size=args.batch_size,
                                            shuffle=False,
                                            num_workers=4)
        test_loader = HSFloader('data/' + args.dataset + '/test.json',
                                        batch_size=args.batch_size,
                                        shuffle=False,
                                        num_workers=4)
    
    else:
        embedding_dict = load_json('data/' + args.dataset + '/emd_dict.json')
        model = LLMeLog(args, embedding_dict).to(args.hard_device)

        train_loader = ADloader('data/' + args.dataset + '/train.txt',
                                            batch_size=args.batch_size,
                                            shuffle=True,
                                            num_workers=4)
        valid_loader = ADloader('data/' + args.dataset + '/dev.txt',
                                            batch_size=args.batch_size,
                                            shuffle=False,
                                            num_workers=4)
        test_loader = ADloader('data/' + args.dataset + '/test.txt',
                                        batch_size=args.batch_size,
                                        shuffle=False,
                                        num_workers=4)

    trainer = pl.Trainer(max_epochs=args.epochs,
                         gpus=None if args.hard_device == 'cpu' else [args.gpu_index],
                         accumulate_grad_batches=args.accumulate_grad_batches)
    
    if args.load_checkpoint:
        model.load_state_dict(torch.load(get_abs_path('checkpoint', f'{model.__class__.__name__}_model.bin'),
                                         map_location=args.hard_device))

    if args.mode == 'eval':
        tp = 0
        tn = 0
        fp = 0
        fn = 0
        acc = 0

        for batch in tqdm(test_loader):
            _, labels = batch
            _, out = model(batch)
            sigmoid = nn.Sigmoid()
            p_labels = sigmoid(out)
            for i in range(len(labels)):
                if (labels[i][1] == 1):
                    if torch.argmax(p_labels[i]) == 1: tp += 1
                    else: fn += 1
                else:
                    if torch.argmax(p_labels[i]) == 1: fp += 1
                    else: tn += 1
        if tp != 0:
            p = tp / (tp + fp)
            r = tp / (tp + fn)
            f1 = 2 * p * r / (p + r)
            acc = (tp + tn) / (tp + tn + fp + fn)
        else:
            p = 0
            r = 0
            f1 = 0
        print("P:", p)
        print("R: ", r)
        print("F1: ", f1) 
        print("Acc: ", acc) 

    elif args.mode == 'gen':
        gen_embedding_dict('data/' + args.dataset)

    elif args.mode == 'train':
        trainer.fit(model, train_loader, valid_loader)

        model.load_state_dict(
            torch.load(get_abs_path('checkpoint', f'{model.__class__.__name__}_model.bin'), map_location=args.hard_device))
        trainer.test(model, test_loader)
    else:
        trainer.test(model, test_loader)


if __name__ == '__main__':
    main()
