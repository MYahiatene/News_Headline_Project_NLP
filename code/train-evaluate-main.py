import tensorflow
from model import build_model, train_data, dev_data, test_data

if __name__ == "__main__":
    train = False
    model = build_model()
    model.summary()
    if train:
        model.fit(train_data, validation_data=dev_data, epochs=3)
        model.save_weights(filepath="./model_weights/weights")
    else:
        model.load_weights(filepath="./model_weights/weights")

    scores = model.evaluate(test_data, return_dict=True)
