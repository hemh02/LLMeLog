import numpy as np
import pandas as pd
import json
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default='hdfs', type=str, help="dataset name")
    return parser.parse_args()

def gen_encoder_data(dataset):
    data = pd.read_csv('data/' + dataset + '/sent.csv')
    row, _ = data.shape
    nomal_dict_list = []
    anomal_dict_list = []
    count = 0
    for i in range(row):
        now_dict = {}
        labels = data.iloc[i, 2]
        if labels == 1: 
            now_dict["src"] = data.iloc[i, 1]
            now_dict["tgt"] = count
            count += 1
            anomal_dict_list.append(now_dict)
        elif labels == 0:
            now_dict["src"] = data.iloc[i, 1]
            now_dict["tgt"] = -1
            nomal_dict_list.append(now_dict)
    if int (len(nomal_dict_list) / len(anomal_dict_list)) != 0:
        all_dict_list = nomal_dict_list + int(len(nomal_dict_list) / len(anomal_dict_list)) * anomal_dict_list
    else:
        all_dict_list = nomal_dict_list + anomal_dict_list

    np.random.shuffle(all_dict_list)
    train = all_dict_list
    test = all_dict_list
    dev = all_dict_list

    test_json = json.dumps(test, ensure_ascii = False, indent = 4)
    with open('data/' + dataset + '/test.json', 'w+', encoding = 'utf-8') as f:
        f.write(test_json)

    dev_json = json.dumps(dev, ensure_ascii = False, indent = 4)
    with open('data/' + dataset + '/dev.json', 'w+', encoding = 'utf-8') as f:
        f.write(dev_json)

    while len(train) < 10000: train += train

    train_json = json.dumps(train, ensure_ascii = False, indent = 4)
    with open('data/' + dataset + '/train.json', 'w+', encoding = 'utf-8') as f:
        f.write(train_json)

def deal_enriched_data(dataset):
    data = pd.read_csv('data/' + dataset + '/enriched.csv') 
    data.insert(loc = 2, column = "LLM_label", value = -1)
    row, _ = data.shape
    for i in (range(row)):
        sent = data.iloc[i, 1]
        sent = json.loads(sent)
        replace_sent = ""
        for key in sent:
            if len(sent) == 1: dic = sent[key]
            else: dic = sent
            for j, item_key in enumerate(dic):
                content = dic[item_key].replace('.', "")
                if j == 0 or j == 1: replace_sent += content + ', '
                if j == 2: replace_sent += content + '.'
                if j == 3: 
                    content = content.lower()
                    if content == "no obvious abnormalities" or content == "unknown": data.iloc[i, 2] = 0
                    elif content == "probably cause anomalies": data.iloc[i, 2] = 1
        data.iloc[i, 1] = replace_sent
    data.to_csv('data/' + dataset + '/sent.csv', index = False)


args = parse_args()
deal_enriched_data(args.dataset)
gen_encoder_data(args.dataset)
