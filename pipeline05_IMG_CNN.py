import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import Scripts.dsutils as dsutils


# def loade_ds():
#     train_images, test_images = train_images / 255.0, test_images / 255.0

#     (train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

#     # Normalize pixel values to be between 0 and 1
#     train_images, test_images = train_images / 255.0, test_images / 255.0

#     class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
#                 'dog', 'frog', 'horse', 'ship', 'truck']
#     return (train_images, train_labels), (test_images, test_labels)
print(tf.__version__)

def load_ds(srcpath, classnames, config):
        ds_train=tf.keras.utils.image_dataset_from_directory(
        srcpath+"/train/",
        color_mode = "grayscale",
        class_names= classnames,
        labels= "inferred",
        seed=230305,
        image_size= image_size,
        batch_size=batchsize)
        ds_train = ds_train.map(lambda x, y: (x / 255.0, y))

        ds_val =tf.keras.utils.image_dataset_from_directory(
        srcpath+"/val/",
        color_mode = "grayscale",
        class_names= classnames,
        labels= "inferred",
        image_size= image_size,
        batch_size=batchsize,
        shuffle=False)
        ds_val = ds_val.map(lambda x, y: (x / 255.0, y))

def create_model():
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(10))
    print(model.summary())
    return model

def train(model, train_images, train_labels, test_images, test_labels):
    model.compile(optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])
    history = model.fit(train_images, train_labels, epochs=10, 
                        validation_data=(test_images, test_labels))

def evaluate(model, test_images, test_labels):
    test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
    print(test_acc)

if __name__ == "__main__":
    load_ds()