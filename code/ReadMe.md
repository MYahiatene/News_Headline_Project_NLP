Python Version: 3.8

Required Python libraries:

-numpy  >=1.21.2

-tensorflow >= 2.6

-keras >= 2.6

-pandas >= 1.3.2

-transformers >= 4.9.2

Alternatively you can use the command "pip install -r requirements.txt" to install the necessary libraries.

train-evaluate-main:
There is a flag in train-evaluate-main.py which is called train. If set to false the model grabs pretrained model from
the ./my_model_base_uncased folder(should be located where the scripts lie) and evaluates on the test set. If set to
true the model trains on the training and dev set and saves the model to ./my_model_base_uncased and evaluates on the
test set.

If the training takes too long I uploaded the model in my google drive. model link:

https://drive.google.com/uc?export=download&id=1lFUEKhw5qmzAKZ-gAS2s8NvRvDLfu0Md

Just unzip the zip in the code folder.

test.py:
The test.py initializes the model with the pretrained moedl and predicts the funnier of two headlines:
The script takes a random sample from the test set and predicts the funnier of both headlines. You can also specify your
own sentences to predict :) .