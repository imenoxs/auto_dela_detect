import yaml
import json
import os
import glob
import shutil
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib import transforms
import scipy
import inspect
import tensorflow as tf
import optuna
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import random


plt.rcParams['image.cmap'] = 'gray'


class pipe_deladetect():
    def __init__(self, trial=None):
        # stores preset parameters
        self.config = None
        self.srcpath = None
        self.dstpath = None
        self.configpath = 'configs/pipeline04config.yaml'
        self.trial = trial
        
        #dataset info
        self.datashape = None
        self.X_mean = None
        self.X_std = None

        # buffers 
        self.raw_image_list = None
        self.currentimagenr = 0
        self.latestimage = None
        self.latestlabel = None
        self.imgrgb = None
        self.imgcrop = None
        # results
        self.eval = None
            
    def crop_img(self):
        bottom = self.config["Cropping"]["bottom"]
        left = self.config["Cropping"]["left"]
        right = self.config["Cropping"]["right"]
        top = self.config["Cropping"]["top"]
        hight,width = self.latestimage.shape
        self.hight=hight-bottom-top
        self.width=width-left-right

        croppedimg = self.latestimage[top:hight-bottom, left:width-right]
        self.latestimage = self.imgcrop = croppedimg

    def load_setup(self): # loads config from config.yaml
        with open(self.configpath, 'r') as file:
            self.config = yaml.safe_load(file)
        self.srcpath = self.config["Paths"]["srcpath"]
        self.dstpath = self.config["Paths"]["dstpath"]
    
    def get_imagepaths(self,fending="RFData.csv", rootpath = None): # loads source images
        if rootpath == None: rootpath = self.srcpath
        image_list = glob.glob(f"{rootpath}/*/*{fending}", recursive=True)
        if image_list == []:
            image_list = glob.glob(f"{rootpath}/*{fending}", recursive=True)
        image_list.sort()
        self.raw_image_list = image_list
    
    def gen_filename(self, path2folder, filename):
        i = 1
        try:
            while os.path.exists(f"{path2folder}/{filename}{i:03d}.png"):
                i += 1
        except: pass
        return f"{path2folder}/{filename}{i:03d}.png"

    def clean_dir(self,path):
        try:
            shutil.rmtree(path)
        except: pass
        os.makedirs(path)

    def trainvaltestsplit(self):
        data_dir = self.srcpath
        split_dir = self.dstpath+"/traindat"
        val_pct = 0.3
        class_dirs = os.listdir(data_dir)
        class_dirs = [x for x in class_dirs if not os.path.isfile(os.path.join(self.srcpath,x))]
        for class_dir in class_dirs:
            class_path = os.path.join(data_dir, class_dir)
            image_filenames = os.listdir(class_path)
            image_paths = [os.path.join(class_path, fn) for fn in image_filenames]
            train_paths, val_paths = train_test_split(image_paths, test_size=val_pct, random_state=23030712)
            train_dir = os.path.join(split_dir, 'train', class_dir)
            val_dir = os.path.join(split_dir, 'val', class_dir)
            self.clean_dir(train_dir)
            self.clean_dir(val_dir)
            for path in train_paths:
                shutil.copy(path, train_dir)
            for path in val_paths:
                shutil.copy(path, val_dir)
            self.srcpath = split_dir
        self.config["General"]["conversionstat"] = True
        self.config["Paths"]["srcpath"] = split_dir
        self.save_config()

    def save_config(self):
        with open(self.configpath, 'w') as outfile:
            yaml.dump(self.config, outfile, default_flow_style=False)

    def create_ds(self):
        x_train = []
        y_train = []
        x_val = []
        y_val = []
        self.datashape = (self.config["General"].get("datahight",0),self.config["General"].get("datawidth",0))
        if 0 not in self.datashape:
            lastinforow = self.datashape[0]
            searchlir = False
        else:
            lastinforow = None
            searchlir = True
        labeldict={
            "normal": 0,
            "delam": 1
        }
        split_dirs = ["train","val"]
        for split_dir in split_dirs:
            if split_dir == "train":
                x_local = x_train
                y_local = y_train
            elif split_dir == "val":
                x_local = x_val
                y_local = y_val
            self.get_imagepaths(rootpath=os.path.join(self.srcpath,split_dir))
            local_datalst = self.raw_image_list
            random.Random(230309114).shuffle(local_datalst)
            for datapath in local_datalst:
                if lastinforow == None and searchlir:
                    for i, lst in enumerate(np.genfromtxt(datapath, delimiter=",")):
                        if max(lst)==0 and searchlir:
                            lastinforow=i-1
                            searchlir = False
                x_local.append(np.genfromtxt(local_datalst[0], delimiter=",")[:lastinforow])
                y_local.append(labeldict[datapath.split("/")[-2]])
        print(x_val[0].shape)


        if self.X_mean == None: self.X_mean = np.mean(x_train)
        if self.X_std == None: self.X_std = np.std(x_train)
        tfds_path=os.path.join(self.dstpath,"tfds")
        self.clean_dir(tfds_path)
        ds_val = self.transform_ds(x_val,y_val)
        tf.data.Dataset.save(ds_val, os.path.join(tfds_path,"val"))
        ds_train = self.transform_ds(x_train,y_train)
        tf.data.Dataset.save(ds_train, os.path.join(tfds_path,"train"))
        self.srcpath = tfds_path

    def transform_ds(self, X, y):
        newshape = X[0].shape
        if 0 in self.datashape:
            self.datashape = newshape
            self.config["General"]["datahight"]=newshape[0]
            self.config["General"]["datawidth"]=newshape[1]
        if newshape != self.datashape:
            raise Exception(f"Shape is not compatibel got shape {newshape} but shape {self.datashape} was expected")
        X = (X-self.X_mean)/self.X_std  
        return tf.data.Dataset.from_tensor_slices((X,y))
    
    
    def create_model(self):
        optimizer = self.config["Hyperparameters"]["optimizer"]
        neurons = self.config["Hyperparameters"]["neurons"]
        layers = self.config["Hyperparameters"]["layers"]        
        lr_adam = self.config["Hyperparameters"]["lr_adam"]

        """         self.model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=self.image_size),  # Flatten the input image
        tf.keras.layers.Dense(neurons, activation='relu'),  # Hidden layer with 128 neurons and ReLU activation
        tf.keras.layers.Dense(1, activation='sigmoid')  # Output layer with 1 neuron and sigmoid activation
        ]) """

        inputs = tf.keras.layers.Input(shape=self.image_size)
        x = tf.keras.layers.Flatten()(inputs)
        x =  tf.keras.layers.Dense(neurons, activation='relu')(x)
        for i in range(layers):
            x =  tf.keras.layers.Dense(neurons, activation='relu')(x)
        outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
        self.model = tf.keras.Model(inputs=inputs, outputs=outputs)

        lossfunctsdict={
            "binary_crossentropy": tf.keras.losses.BinaryCrossentropy(from_logits=False),
        }
        optimizerdict={
            "adam" : tf.keras.optimizers.Adam(learning_rate=lr_adam),
            }
        self.model.compile(optimizer=optimizerdict[optimizer], loss=lossfunctsdict["binary_crossentropy"], metrics=['accuracy'])
    
    def trainmodel(self):
        print(self.config["Hyperparameters"])
        callbackslst=[]
        cb_earlystop= tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience= 5,
                restore_best_weights=True,
                start_from_epoch=5
                )
        callbackslst.append(cb_earlystop)

        cb_checkpoint = tf.keras.callbacks.ModelCheckpoint(
                self.srcpath+"/model/epoche_{epoch:02d}-acc_{val_accuracy:.2f}-loss_{val_loss:.2f}.hdf5",
                save_best_only= True)
        if self.trial ==  None: callbackslst.append(cb_checkpoint)

        cb_lambda =  tf.keras.callbacks.LambdaCallback(
            on_epoch_end= lambda epoch, logs: self.pruning(epoch,logs)
        )
        if self.trial != None: callbackslst.append(cb_lambda)

        epochs = self.config["Hyperparameters"]["epochs"]
        self.history = self.model.fit(self.ds_train,validation_data=self.ds_val ,epochs = epochs, verbose = 1, callbacks=callbackslst)

    def pruning(self, step, logs):
        objectiveval=logs["val_accuracy"]
        self.trial.report(objectiveval, step)
        if self.trial.should_prune():
            raise optuna.TrialPruned()

    def run_pipeline(self):
        self.load_setup()
        if self.config["General"]["conversionstat"] == False:
            self.get_imagepaths(fending="RFData.csv")
            self.trainvaltestsplit()
        self.create_ds()
        #self.create_model()
        #self.trainmodel()
        #self.gen_confmatrix()
        self.save_config()

    def gen_confmatrix(self):
        ds = self.ds_val
        y_pred = self.model.predict(ds)
        y_pred_classes = y_pred.round().astype("int")
        y_true_classes = np.concatenate([y for x, y in ds], axis=0)
        conf_matrix = confusion_matrix(y_true_classes, y_pred.round().astype("int"))
        self.acc= (conf_matrix[1][1]+conf_matrix[0][0])/len(y_pred_classes)

        fig, ax = plt.subplots(figsize=(7.5, 7.5))
        ax.matshow(conf_matrix, cmap=plt.cm.Blues, alpha=0.3)
        for i in range(len(conf_matrix)):
            for j in range(len(conf_matrix[1])):
                ax.text(x=j, y=i,s=conf_matrix[i][j], va='center', ha='center', size='xx-large')
        plt.xlabel('Predictions', fontsize=18)
        plt.ylabel('Actuals', fontsize=18)
        plt.title(f'Confusion Matrix', fontsize=18)
        plt.figtext(0, 0, f'Dataset Size: {len(y_pred_classes)}\nSample Split: {sum(conf_matrix[0]/sum(conf_matrix[1]))}\nAcc: {self.acc}\nOptimizer: {self.config["Hyperparameters"]["optimizer"]}')
        fig.savefig(self.dstpath+"/confusion")
        plt.close("all")
        print("Final val_acc: " + str(self.acc))


if __name__ == "__main__":
    new_pipeline = pipe_deladetect()
    new_pipeline.run_pipeline()
