import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import utils
from transformers import BertTokenizer
import transformers

bert_model = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(bert_model)
max_length = 256
batch_size = 8


class data_sets():
    def __init__(self, tokenizer: BertTokenizer, max_len: int = max_length):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.train, self.test, self.dev = self._import_corpus_()
        self.data_set_train, self.data_set_dev, self.data_set_test = self._prepare_data_()

    def _import_corpus_(self):
        test = pd.read_csv("test.csv")
        train = pd.read_csv("train.csv")
        dev = pd.read_csv("dev.csv")
        complete_set_len = len(train) + len(dev) + len(test)
        train_percent = round(len(train) / complete_set_len * 100)
        dev_percent = round(len(dev) / complete_set_len * 100)
        test_percent = round(len(test) / complete_set_len * 100)
        print("train data:", len(train), "({}%)".format(train_percent))
        print("validation data:", len(dev), "({}%)".format(dev_percent))
        print("test set:", len(test), "({}%)".format(test_percent))
        return train, test, dev

    def _prepare_data_(self):
        input_ids_train, attention_masks_train, labels_train = self._prepare_(self.train)
        input_ids_dev, attention_masks_dev, labels_dev = self._prepare_(self.dev)
        input_ids_test, attention_masks_test, labels_test = self._prepare_(self.test)
        data_set_train = tf.data.Dataset.from_tensor_slices(
            ({"input": input_ids_train, "mask": attention_masks_train}, labels_train))
        data_set_dev = tf.data.Dataset.from_tensor_slices(
            ({"input": input_ids_dev, "mask": attention_masks_dev}, labels_dev))
        data_set_test = tf.data.Dataset.from_tensor_slices(
            ({"input": input_ids_test, "mask": attention_masks_test}, labels_test))
        return data_set_train, data_set_dev, data_set_test

    def _prepare_(self, data):
        attention_arr = np.zeros((len(data), self.max_len))
        input_ids_arr = np.zeros((len(data), self.max_len))
        for i in range(len(data)):
            row = data.iloc[i]
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
        labels = data["label"].tolist()
        labels = utils.to_categorical(labels, num_classes=len(set(labels)))
        return input_ids_arr, attention_arr, labels
