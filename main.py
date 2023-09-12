import torch
import torch.nn as nn
import torchtext
import spacy
import pandas as pd
import Model
from torch.utils.data import DataLoader
import json

if __name__ == '__main__':
    label_list = ['民生 故事', '文化 文化',
                  '娱乐 娱乐', '体育 体育',
                  '财经 财经', '',
                  '房产 房产', '汽车 汽车',
                  '教育 教育', '科技 科技',
                  '军事 军事', '',
                  '旅游 旅游', '国际 国际',
                  '证券 股票', '农业 三农',
                  '电竞 游戏']
    # sen_size = 20
    df = pd.read_csv("./info.csv")
    with open("str_to_int_vocab.json", "r", encoding="utf-8") as f:
        vocab = json.loads(f.read())
    # with open("cut.json", "r", encoding="utf-8") as f:
    #     text_list = json.loads(f.read())["data"]
    # train_iter = Model.DataFrameDataset(text_list[:190000], list(df['category'])[:190000])
    # valid_iter = Model.DataFrameDataset(text_list[190000:], list(df['category'])[190000:])
    # print(len(train_iter))
    # train_loader = DataLoader(train_iter, batch_size=16, shuffle=True
    #                           ,collate_fn=Model.collate_batch)
    # valid_loader = DataLoader(valid_iter, batch_size=16, shuffle=True
    #                           ,collate_fn=Model.collate_batch)
    # for i in train_loader:
    #     print(i)
    #     break
    # print(len(vocab))
    model = Model.LstmNet(hidden_size=100, embedding_dim=300, vocab_size=len(vocab))
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    loss_fn = torch.nn.CrossEntropyLoss()
    param_path="./params/params.pth"
    epochs = 20
    lstm = Model.LstRun(epochs=epochs,model=model,optimizer=optimizer,loss_fn=loss_fn,
                        param_path=param_path,labels=label_list)
    # lstm.train(train_loader,valid_loader)
    sentence = "美联储或将决定降息2%"
    sentence_convert = Model.convert_sentence(sentence, vocab, 20)
    print(sentence)
    print(lstm.predict(sentence=sentence_convert, top=3))



