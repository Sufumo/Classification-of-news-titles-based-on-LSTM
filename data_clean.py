import pandas as pd
import torchtext
import pkuseg
import torch
from torch.utils.data import Dataset, DataLoader
import functools
import json
import time
seg = pkuseg.pkuseg()


class DataFrameDataset(Dataset):
    def __init__(self, content, label):
        self.content = content
        self.label = label

    def __getitem__(self, index):
        return self.content[index], self.label[index]

    def __len__(self):
        return len(self.label)


def collate_batch(data):
    unit_x = []
    unit_y = []
    for unit in data:
        unit_x.append(unit[0])
        unit_y.append(unit[1]-101)
    return {"text": torch.tensor(unit_x),
            "label": torch.tensor(unit_y)}


def yield_tokens(data_df):
    for _, row in data_df.iterrows():
        yield [token for token in seg.cut(row.title)]


if __name__ == '__main__':
    label_list = ['民生 故事', '文化 文化',
                  '娱乐 娱乐', '体育 体育',
                  '财经 财经', '房产 房产',
                  '汽车 汽车', '教育 教育',
                  '科技 科技', '军事 军事',
                  '旅游 旅游', '国际 国际',
                  '证券 股票', '农业 三农',
                  '电竞 游戏']
    # sen_size = 20
    df = pd.read_csv("./info.csv")
    with open("int_to_str_vocab.json", "r", encoding="utf-8") as f:
        vocab = json.loads(f.read())
    with open("cut.json", "r", encoding="utf-8") as f:
        text_list = json.loads(f.read())["data"]
    train_iter = DataFrameDataset(text_list, list(df['category']))
    train_loader = DataLoader(train_iter, batch_size=8, shuffle=True
                              ,collate_fn=collate_batch)
    # for loader in train_loader:
    #     text = loader["text"]
    #     label = loader["label"]
    #     for e,t in enumerate(text):
    #         for i,j in enumerate(t):
    #             t[i] = vocab[f"{j}"]
    #         print("".join(t))
    #         print(label_list[label[e]])
    #     break
    # li1 = []
    # for i in df.index:
    #     temp = seg.cut(df.loc[i,"title"])
    #     len1 = len(temp)
    #     if len1 > sen_size:
    #         temp =temp[0:sen_size]
    #     else:
    #         temp.extend(["<pad>"]*(sen_size - len1))
    #     for e,j in enumerate(temp):
    #         if j in vocab.keys():
    #             temp[e] = vocab[j]
    #         else:
    #             temp[e] = 0
    #     li1.append(temp)
    #     temp = []
    # with open("cut.json", "r", encoding="utf-8") as f:
        # li1 = json.loads(f.read())["data"][100]
        # for i,j in enumerate(li1):
        #     li1[i] = vocab[f"{j}"]
        # print("".join(li1))


