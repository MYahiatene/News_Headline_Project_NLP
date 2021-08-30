Required Python libraries:
-numpy -tensorflow -keras -pandas -transformers

train-evaluate-main>
There is a flag in train-evaluate-main.py which is called train. If set to false the model grabs pretrained model from
the ./my_model_base_uncased folder(should be located where the scripts lie) and evaluates on the test set. If set to
true the model trains on the training and dev set and saves the model to ./my_model_base_uncased and evaluates on the
test set.

If the training takes too long I uploaded the model in my google drive. model link:

https://drive.google.com/drive/folders/1EPxJ4PwoBt-AIFfNVTzed7vlvnxmGiZb?usp=sharing

test.py:
The test.py initializes the model with the pretrained moedl and predicts the funnier of two headlines:
The script takes a random sample from the test set and predicts the funnier of both headlines. You can also specify your
own sentences to predict :) .