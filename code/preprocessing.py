import pandas as pd
import transformers
import tensorflow
from transformers import BertTokenizer
#original1,edit1,original2,edit2
bert_model = "bert-base-cased"

def import_corpus():
    path = "../Paper/semeval-2020-task-7-dataset/subtask-2/"
    test = pd.read_csv(path + "test.csv",usecols=["original1","edit1","original2","edit2"])
    train = pd.read_csv(path + "train.csv",usecols=["original1","edit1","original2","edit2"])
    dev = pd.read_csv(path + "dev.csv",usecols=["original1","edit1","original2","edit2"])
    return train, test, dev


train, test, dev = import_corpus()
for _,row in train.iterrows():
    print(row)
tokenizer=BertTokenizer.from_pretrained(bert_model)
print(train["original2"][0])
print(train["edit2"][0])
print(tokenizer.encode_plus(train["original2"][0],max_length=130,padding="max_length"))
print(tokenizer.decode(tokenizer(train["original1"][0],train["original2"][0])["input_ids"]))
print(tokenizer(train["original1"][0],train["original2"][0],return_tensors="tf"))