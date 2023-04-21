import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import os


# def loade_ds():
#     train_images, test_images = train_images / 255.0, test_images / 255.0

#     (train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

#     # Normalize pixel values to be between 0 and 1
#     train_images, test_images = train_images / 255.0, test_images / 255.0

#     class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
#                 'dog', 'frog', 'horse', 'ship', 'truck']
#     return (train_images, train_labels), (test_images, test_labels)
print(tf.__version__)

def load_ds(srcpath, batch_size, image_size):

    #load data into train and val dataset
    ds_train, ds_val=tf.keras.utils.image_dataset_from_directory(
    directory=srcpath,
    validation_split=0.3,
    subset="both",
    color_mode = "grayscale",
    # class_names= ["defect","nodefect"],
    labels= "inferred",
    seed=230421,
    image_size= image_size,
    batch_size=batch_size)
    ds_train = ds_train.map(lambda x, y: (x / 255.0, y))
    ds_val = ds_val.map(lambda x, y: (x / 255.0, y))
    return ds_train, ds_val

def create_model(input_size):
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(input_size[0], input_size[1], 1)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(10))
    print(model.summary())
    return model

def train(model, train_images, test_images):
    model.compile(optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])
    history = model.fit(train_images, epochs=10, 
                        validation_data=(test_images))

def evaluate(model, test_images, test_labels):
    test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
    print(test_acc)

if __name__ == "__main__":
    srcpath = "dst/2303_pez500/png"
    batchsize = 32


    #get one image to extract image dimensions
    for dirpath, _, filenames in os.walk(srcpath):
        for filename in filenames:
            with tf.keras.preprocessing.image.load_img(os.path.join(dirpath, filename)) as img:
                image_size = (img.height, img.width)
            break
    ds_train, ds_val = load_ds(srcpath, batch_size=batchsize, image_size=image_size)
    model = create_model(input_size=image_size)
    train(model, ds_train, ds_val)