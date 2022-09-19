from my_tokenizer_2 import constants
from my_tokenizer_2 import tokenize
from my_tokenizer_2 import classify
from csv import reader
import torch
import numpy as np
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def collate_batch(batch):
    label_list, text_list, offsets = [], [], [0]
    for (_label, _text) in batch:
        #processed_text = torch.tensor(_text, dtype=torch.int64) #----------RESTORE
        _text = np.array(_text)
        processed_text = torch.tensor(_text , dtype=torch.float32)
        text_list.append(processed_text)
        offsets.append(processed_text.size(0))
        #processed_label = torch.tensor(_label, dtype=torch.int64) #----------RESTORE
        _label = np.array(_label)
        processed_label = torch.tensor(_label , dtype=torch.float32)
        label_list.append(processed_label)
    offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
    text_list = torch.cat(text_list)
    #label_list = torch.tensor(label_list, dtype=torch.int64) #----------RESTORE
    label_list = torch.tensor(label_list, dtype=torch.float32)
    return label_list.to(device), text_list.to(device), offsets.to(device)



def prepare_data(path_to_file):
    csv_data = []
    train_iter = []
    print(len(constants.TOKENS))
    print(constants.TOKENS[1])
    print(constants.TOKENS.index('d'))
    with open(path_to_file, 'r') as read_obj:
        csv_reader = reader(read_obj)
        for row in csv_reader:
            csv_data.append(row)
    for data in csv_data:
        train_iter.append([classify(data[1]),tokenize(data[0])]) #----------RESTORE
    return train_iter

if __name__ == '__main__':
    prepare_data(constants.PATH_TO_TRAIN_DATA)