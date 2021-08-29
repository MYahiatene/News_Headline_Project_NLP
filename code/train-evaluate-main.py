import tensorflow
from model import build_model, train_data, dev_data, test_data

if __name__ == "__main__":
    train = False
    if train:
        model = build_model()
        model.summary()
        model.fit(train_data, validation_data=dev_data, epochs=3)
        model.save("./my_model_base_uncased")
    else:
        model = tensorflow.keras.models.load_model('saved_model/my_model')
        model.summary()

    scores = model.evaluate(test_data, return_dict=True)
