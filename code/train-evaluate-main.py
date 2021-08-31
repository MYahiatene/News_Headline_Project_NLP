import tensorflow
from model import build_model, train_data, dev_data, test_data

### Our main script file to train and evaluate our model ###
if __name__ == "__main__":
    train = True  # if set to false we load our saved model and don't train, else we build new model and train.
    if train:
        model = build_model()
        model.summary()
        model.fit(train_data, validation_data=dev_data, epochs=3)
        model.save("./my_model_base_uncased")
    else:
        model = tensorflow.keras.models.load_model("./my_model_base_uncased")
        model.summary()

    scores = model.evaluate(test_data, return_dict=True)
