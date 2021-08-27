Required Python libraries:
-numpy
-tensorflow
-keras
-pandas
-transformers

There is a flag in train-evaluate-main.py which is called train.
If set to false the model grabs pretrained weights from the model_weights folder(should be located where the scripts lie).
If set to true the model trains on the training and dev set and saves the weights in the model_weights folder.

If the training takes too long I uploaded the weights in my google drive.
Weights link: https://drive.google.com/drive/folders/1-DTjj862s7HH14KfvnH5y6kbZ6nX8tQs?usp=sharing


test.py:
The test.py initializes the model with the pretrained weights and predicts the funnier of two headlines: