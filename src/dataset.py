from torch.utils.data import Dataset, DataLoader
import os
import json
import numpy as np
import torch

def load_json(fp):
    if not os.path.exists(fp):
        return dict()

    with open(fp, 'r', encoding='utf8') as f:
        return json.load(f)

class ADdatasets(Dataset):
    def __init__(self, fp):
        with open(fp, 'r') as f:
            self.data = f.readlines()

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        sent = self.data[index].replace('\n', '')
        sent = sent.split(':')
        label = int(sent[0])
        src = sent[1]
        return src, label          


def padding_collate(batch):  
    src = []
    label = []

    max_len = -1
    for item in batch:
        sent = item[0].split(' ')
        src.append(sent)
        if len(sent) > max_len: max_len = len(sent)

        if int(item[1]) == 0: label.append([1, 0])
        else: label.append([0, 1])        

    for i, item in enumerate(src):
        j = max_len - len(item)
        src[i].extend([0] * j)
    return src, label


def ADloader(fp, batch_size = 4, shuffle = True, num_workers = 0):
    dataset = ADdatasets(fp)
    loader = DataLoader(dataset,batch_size= batch_size, shuffle = shuffle,num_workers= num_workers, collate_fn= padding_collate)
    return loader


class HSFdatasets(Dataset):
    def __init__(self, fp):
        self.data = load_json(fp)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.data[index]["src"], self.data[index]["tgt"]


def HSFcollate(batch):  
    nomal_src = []
    anomal_src = []
    nomal_label = []
    anomal_label = []
    anomal_class_label = []

    for item in batch:
        sent = item[0]
        sent_label = item[1]
        if sent_label == -1:
            nomal_label.append([1.0, 0.0])
            nomal_src.append(sent)
        else: 
            anomal_label.append([0.0, 1.0])
            anomal_src.append(sent)
            anomal_class_label.append(sent_label)
            
        
    
    nomal_label = np.array(nomal_label)
    nomal_label = torch.from_numpy(nomal_label)

    anomal_label = np.array(anomal_label)
    anomal_label = torch.from_numpy(anomal_label)

    anomal_class_label = np.array(anomal_class_label).astype(np.int64)
    anomal_class_label = torch.from_numpy(anomal_class_label)

    return nomal_src, anomal_src, nomal_label, anomal_label, anomal_class_label


def HSFloader(fp, batch_size = 4, shuffle = True, num_workers = 0):
    dataset = HSFdatasets(fp)
    loader = DataLoader(dataset,batch_size= batch_size, shuffle = shuffle,num_workers= num_workers, collate_fn= HSFcollate)
    return loader