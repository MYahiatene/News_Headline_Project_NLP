import transformers
import tensorflow
import numpy as np
from preprocessing import max_length
from model import build_model

model = build_model()
model.load_weights(filepath="./model_weights/weights")


def predict(original, edit1, edit2):
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
          round(prediction_softmax[0][0]*100), "\nEdit1 funnier than Edit2: ", round(prediction_softmax[0][1]*100),
          "\nEdit2 funnier than Edit1: ", round(prediction_softmax[0][2]*100))
    prediction = np.argmax(prediction_softmax)
    print("Edit {} is funnier! :)".format(prediction))
    return np.argmax(prediction_softmax)


original = "40 percent of voters believe Trump is fit to be president , a new low "
edit1 = "40 percent of gnomes believe Trump is fit to be president , a new low "
edit2 = "40 percent of voters believe Trump is fit to be triathlete , a new low "
predicted_label = predict(original, edit1, edit2)
