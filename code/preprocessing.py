import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import utils
from transformers import BertTokenizer

bert_model = "bert-base-cased"
tokenizer = BertTokenizer.from_pretrained(bert_model)
max_length = 512

def prepare_data():
    train, test, dev = import_corpus()
    labels_train = utils.to_categorical(
        labels := train["meanGrade1"].append(train["meanGrade2"], ignore_index=True).values,
        num_classes=len(set(labels)))
    labels_dev = utils.to_categorical(labels := dev["meanGrade1"].append(dev["meanGrade2"], ignore_index=True).values,
                                      num_classes=len(set(labels)))
    data_train = data_set(train, tokenizer, max_length)
    input_ids_train, attention_masks_train, labels_train = data_train.prepare()
    labels_train = utils.to_categorical(labels_train, num_classes=3)
    labels_dev = utils.to_categorical(labels_dev, num_classes=3)
 #   data_test = data_set(test, tokenizer, max_length)
#    input_ids_test, attention_masks_test, labels_test = data_test.prepare()
    data_dev = data_set(dev, tokenizer, max_length)
    input_ids_dev, attention_masks_dev, labels_dev = data_dev.prepare()
    data_set_train = tf.data.Dataset.from_tensor_slices(({"input":input_ids_train, "mask":attention_masks_train}, labels_train))
    data_set_dev = tf.data.Dataset.from_tensor_slices(({"input":input_ids_dev, "mask":attention_masks_dev}, labels_dev))
    return data_set_train, data_set_dev


def import_corpus():
    test = pd.read_csv("test.csv")
    train = pd.read_csv("train.csv")
    dev = pd.read_csv("dev.csv")
    return train, test, dev


class data_set():
    def __init__(self, data: pd.DataFrame, tokenizer: BertTokenizer, max_len: int = 128):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def prepare(self):
        attention_arr = np.zeros((self.__len__(), self.max_len))
        input_ids_arr = np.zeros((self.__len__(), self.max_len))
        labels = list()
        for i in range(self.__len__()):
            row = self.data.iloc[i]
            text_original1 = row["original1"].replace("<", "").replace("/>", "")
            text_edited1 = row["original1"].split("<")[0] + row["edit1"] + \
                           row["original1"].split(">")[1]
            text_original2 = row["original2"].replace("<", "").replace("/>", "")
            text_edited2 = row["original2"].split("<")[0] + row["edit2"] + \
                           row["original2"].split(">")[1]
            batch = self.tokenizer.encode_plus(
                text_original1 + " " + text_edited1 + " " + text_original2 + " " + text_edited2,
                add_special_tokens=True,
                max_length=self.max_len,
                return_token_type_ids=False, padding="max_length", truncation=True,
                return_attention_mask=True, return_tensors="tf")
            input_ids_arr[i, :] = batch["input_ids"]
            attention_arr[i, :] = batch["attention_mask"]
            if row["meanGrade1"] - row["meanGrade2"] < 0:
                labels.append(2)
            elif row["meanGrade1"] - row["meanGrade2"] > 0:
                labels.append(1)
            else:
                labels.append(0)
        return input_ids_arr, attention_arr, labels
