import torch
import torchdata.datapipes as dp
from torch.utils.data import DataLoader
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

tokenizer = get_tokenizer('basic_english')

def yield_tokens(data_iter):
    for text in data_iter:
        yield tokenizer(text)

def filter_for_data(filename):
    return "train" in filename and filename.endswith(".csv")

def row_processer(row):
    return {"label": row[1], "data": row[0]}

def build_datapipes(root_dir="./data/"):
    datapipe = dp.iter.FileLister(root_dir)
    datapipe = datapipe.filter(filter_fn=filter_for_data)
    datapipe = datapipe.open_files(mode='rt')
    datapipe = datapipe.parse_csv(delimiter=",", skip_lines=1)
    datapipe = datapipe.map(row_processer)
    datapipe = datapipe.sharding_filter()
    return datapipe


if __name__ == '__main__':
    datapipe = build_datapipes()
    dl = DataLoader(dataset=datapipe, batch_size=5, num_workers=2, shuffle=True)
    first = next(iter(dl))
    labels, features = first['label'], first['data']
    print(f"{labels }\n{features }")
    n_sample = 0
    for row in iter(dl):
        n_sample += 1
    print(f"{n_sample }")

