import transformers
import tensorflow as tf
import numpy as np
from preprocessing import max_length
from model import data_set

model = tf.keras.models.load_model("./my_model_base_uncased")


# Grabs a random sample from our test data set
def grab_random_sample():
    '''
    @return:  originial sentence, edited sentence 1, edited sentence 2
    @rtype: string, string, string
    '''
    test_data = data_set.test
    rnd = np.random.randint(0, len(test_data))
    row = test_data.iloc[rnd]
    original = row["original1"].replace("<", "").replace("/>", "")
    text_edited1 = row["original1"].split("<")[0] + row["edit1"] + \
                   row["original1"].split(">")[1]
    text_edited2 = row["original2"].split("<")[0] + row["edit2"] + \
                   row["original2"].split(">")[1]
    return original, text_edited1, text_edited2


# predicts which edited sentence is funnier
def predict(original, edit1, edit2):
    '''
    :param original: original sentence
    :type original: string
    :param edit1: edited sentence 1
    :type edit1: string
    :param edit2: edited sentence 2
    :type edit2: string
    :return: a label 0,1 or 2
    :rtype: int
    '''
    original = original
    edit1 = edit1
    edit2 = edit2
    text = original + edit1 + original + edit2
    tokens = transformers.BertTokenizer.from_pretrained("bert-base-cased").encode_plus(text=text, max_length=max_length,
                                                                                       truncation=True,
                                                                                       padding='max_length',
                                                                                       add_special_tokens=True,
                                                                                       return_token_type_ids=False,
                                                                                       return_tensors='tf')
    prediction_softmax = model.predict(dict(input=tokens["input_ids"], mask=tokens["attention_mask"]))
    print("The original sentence is: ", original)
    print("The first edited sentence is: ", edit1)
    print("The second edited sentence is: ", edit2)
    print("Probabilities for the sentences to be equally funny or one of them being funnier:", "\nEqually funny: ",
          round(prediction_softmax[0][0] * 100), "%", "\nEdit1 funnier than Edit2: ",
          round(prediction_softmax[0][1] * 100), "%",
          "\nEdit2 funnier than Edit1: ", round(prediction_softmax[0][2] * 100), "%")
    prediction = np.argmax(prediction_softmax)
    print("Edit {} is funnier! :)".format(prediction))
    return np.argmax(prediction_softmax)


original, edit1, edit2 = grab_random_sample()
# We can specify our own sentences if we dont want to grab random ones from test data:
'''original = "40 percent of voters believe Trump is fit to be president , a new low "
edit1 = "40 percent of gnomes believe Trump is fit to be president , a new low "
edit2 = "40 percent of voters believe Trump is fit to be triathlete , a new low "'''
predicted_label = predict(original, edit1, edit2)
