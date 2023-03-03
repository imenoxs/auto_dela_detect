import yaml
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
plt.rcParams['image.cmap'] = 'gray'


class pipe_deladetect():
    def __init__(self):
        # stores preset parameters
        self.config = None
        self.srcpath = None
        self.dstpath = None

        # keeping track of which processes were already run
        self.status = {}

        # buffers 
        self.raw_image_list = None
        self.currentimagenr = 0
        self.latestimage = None
        self.imgrgb = None
        self.imgcrop = None
        # results
        self.eval = None
        self.predictions = pd.DataFrame(columns=["filename","peaks","prediction", "criteria"])

    def load_new_image(self, imgpath):
        latestimage = cv2.imread(imgpath)
        self.latestimage = cv2.cvtColor(latestimage, cv2.COLOR_BGR2GRAY)
        self.imgrgb = cv2.cvtColor(latestimage, cv2.COLOR_BGR2RGB)
        
    def crop_img(self):
        bottom = self.config["Cropping"]["bottom"]
        left = self.config["Cropping"]["left"]
        right = self.config["Cropping"]["right"]
        top = self.config["Cropping"]["top"]
        hight,width = self.latestimage.shape

        croppedimg = self.latestimage[top:hight-bottom, left:width-right]
        self.latestimage = self.imgcrop = croppedimg

    def thr_image(self):
        thrval = self.config["Thresholding"]["thrval"]
        _, threshold = cv2.threshold(self.latestimage, thrval, 255, cv2.THRESH_BINARY)
        self.latestimage = threshold

    def ero_dil(self):
        kernel = np.array(
                          [[0.,0.,0.],
                          [1.,1.,1.],
                          [0.,0.,0.]],np.uint8)
        img_thr = self.latestimage
        ero_iters=self.config["Erodil"]["ero_iters"]
        dil_iters = self.config["Erodil"]["dil_iters"]
        img_erosion = cv2.erode(img_thr, kernel, iterations=ero_iters)
        img_dilation = cv2.dilate(img_erosion, kernel, iterations=dil_iters)
        self.latestimage = img_dilation

    def analyse(self):
        df = pd.DataFrame(self.latestimage)
        colsum=df.sum(axis=1) #adding up columns to have the distribution of white pixels over the rows
        max=colsum.max()
        filterval = self.config["Processing1"]["threshold"]
        moving_average = self.config["Processing1"]["movingaverage"]
        # some smoothing has to be applied to filter out noise
        colsum = pd.Series(colsum.rolling(moving_average).mean())
        colsum = colsum.apply(lambda x: 0 if x <= max/100*filterval or np.isnan(x) else x)
        peaks, props = scipy.signal.find_peaks(colsum, height=0)
        
        if False:
            aspectratio = "auto"
            fig, axs = plt.subplots(1,4,figsize=(10,5), sharey=True)
            bases=[]
            for i in range(len (axs)):
                bases.append(axs[i].transData)
            rot = transforms.Affine2D().rotate_deg(90)
            axs[0].imshow(self.imgcrop, aspect = aspectratio)
            axs[0].title.set_text(f"Original Image")
            axs[1].imshow(self.latestimage, aspect = aspectratio)
            axs[1].title.set_text(f"B/W Image")
            
            axs[2].plot(colsum,transform= rot + bases[2])
            axs[2].title.set_text(f"Accum Pixel Val\nHightThrPerc={filterval}%")
            axs[2].invert_xaxis()
            axs[3].plot(colsum,transform= rot + bases[3])
            axs[3].plot(peaks, colsum[peaks], "x", transform= rot + bases[3])
            axs[3].title.set_text(f"Local Maxima = {len(peaks)}")
            axs[3].invert_xaxis()
            for ax in axs:
                ax.tick_params(labelrotation=45)
            # plt.show()
            savedir = self.dstpath+"/processed"
            fname= self.gen_filename(savedir,"overview")
            fig.savefig(fname)
            plt.close('all')

        predictfunc = lambda peaks: True if len(peaks)>2 else False
        funcString = str(inspect.getsourcelines(predictfunc)[0])
        funcString = funcString.strip("['\\n']").split(" = ")[1]
        prediction = predictfunc(peaks)

        pred = {"filename": self.raw_image_list[self.currentimagenr],
                "peaks": len(peaks),
                "prediction": prediction,
                "criteria": funcString}

        self.predictions.loc[len(self.predictions)] = pred


    def convert_csv2png(self):
        convdstpath = f"{self.dstpath}/png/"
        self.get_imagepaths(fending="Data.csv")
        self.clean_dir(convdstpath)
        for imgpath in self.raw_image_list:
            img=pd.read_csv(imgpath).to_numpy()
            fname= self.gen_filename(convdstpath,"ImagePData")
            _=plt.imsave(fname,img)
        self.srcpath = self.dstpath+"/png"
        self.get_imagepaths()



    def load_setup(self): # loads config from config.yaml
        with open('config.yaml', 'r') as file:
            self.config = yaml.safe_load(file)
        self.srcpath = self.config["Paths"]["srcpath"]
        self.dstpath = self.config["Paths"]["dstpath"]
    
    def get_imagepaths(self,fending=".png"): # loads source images
        image_list = glob.glob(f"{self.srcpath}/*{fending}", recursive=True)
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
    
    def show_image(self):
        cv2.imshow("test", self.latestimage)
        cv2.waitKey(0)

    def save_preds(self):
        self.predictions.to_csv(self.dstpath+"/prediction.csv")

    def one_cycle(self):
        self.load_new_image(self.raw_image_list[self.currentimagenr])
        self.crop_img()
        self.thr_image()
        self.ero_dil()
        self.analyse()
        self.currentimagenr += 1
    
    def load_eval(self):
        evalpath = self.srcpath+"/labels.csv"
        if os.path.exists(evalpath):
            self.eval = pd.read_csv(evalpath)

    def run_pipeline(self):
        self.load_setup()
        self.load_eval()
        self.get_imagepaths()
        self.convert_csv2png()
        self.clean_dir(self.dstpath+"/processed")
        try:
            while True: 
                self.one_cycle()
        except: pass
        self.save_preds()
        self.gen_confmatrix()
        #self.show_image()
    
    def gen_confmatrix(self):
        preds= self.predictions[["filename","prediction"]]
        df=self.eval
        counttrue = len(df[df["label"]==1])
        countfalse = len(df[df["label"]==0])
        df["preds"]=preds["prediction"].apply(lambda x: 1 if x == True else 0)

        matrix = {}
        for group, frame in df[["preds","label"]].groupby("label"):
            matrix[group]=dict(frame.value_counts().reset_index().set_index("preds")[0])
        confusionMatrix =[[matrix[1][1]/counttrue,matrix[1][0]/counttrue], [matrix[0][1]/countfalse,matrix[0][0]/countfalse]]

        acc_overall= (matrix[1][1]+matrix[0][0])/len(df)

        fig, ax = plt.subplots(figsize=(7.5, 7.5))
        conf_matrix=confusionMatrix
        ax.matshow(conf_matrix, cmap=plt.cm.Blues, alpha=0.3)
        for i in range(len(conf_matrix)):
            for j in range(len(conf_matrix[1])):
                ax.text(x=j, y=i,s=conf_matrix[i][j], va='center', ha='center', size='xx-large')
        
        plt.xlabel('Predictions', fontsize=18)
        plt.ylabel('Actuals', fontsize=18)
        plt.title(f'Confusion Matrix', fontsize=18)
        plt.figtext(0, 0, f'Dataset Size: {len(df)}\nSample Split: {countfalse/len(df)}\nAcc: {acc_overall}')
        #plt.show()
        fig.savefig(self.dstpath+"/confusion")
        plt.close("all")


if __name__ == "__main__":
    new_pipeline = pipe_deladetect()
    new_pipeline.run_pipeline()
