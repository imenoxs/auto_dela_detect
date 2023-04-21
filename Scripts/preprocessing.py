import dsutils
import os
import pandas as pd
import matplotlib.pyplot as plt

def main(dstpath, labels_df,cut_vals, fname_schema):
    convert_csv2png(dstpath=dstpath,labels_df=labels_df, cut_vals=cut_vals)

def convert_csv2png(dstpath, labels_df, cut_vals):
    convdstpath = f"{dstpath}/png/" #destination to save converted images
    dsutils.clean_dir(convdstpath) #if not exists creates destination path or else deletes everything in it
    for index, row in labels_df.iterrows():
        imgpath = row["paths"]
        label = row["labels"]
        img=pd.read_csv(imgpath).to_numpy()
        cropped_img = crop_img(img=img, cut_vals=cut_vals)

        #generate new saving path
        fdir,orig_fname = os.path.split(imgpath)
        new_fname = orig_fname.split(".")[0]
        savepath=os.path.join(convdstpath,label,new_fname+".png")
     
        if not os.path.exists("/".join(savepath.split("/")[:-1])):
             dsutils.clean_dir("/".join(savepath.split("/")[:-1]))
        _=plt.imsave(savepath,cropped_img, cmap='gray')

def crop_img(img, cut_vals):
    hight,width = img.shape
    croppedimg = img[cut_vals["top"]:hight-cut_vals["bottom"], cut_vals["left"]:width-cut_vals["right"]]
    return croppedimg


if __name__ == "__main__":
    configpath = os.path.join("configs","pipeline06config.yaml")
    labels_df_path = os.path.join("src","2303_pez500","2303_pez500_labels.csv")
    dstpath = os.path.join("dst","test")
    srcpath = os.path.join("src","2303_pez500")
    fname_schema="cdata*.csv"
    cut_vals={"bottom":70,
            "left":0,
            "top":0,
            "right":2}
    
    cwd = os.path.split(os.path.dirname(os.path.abspath(__file__)))[0]
    labels_df = pd.read_csv(labels_df_path, index_col=0)
    #config = dsutils.load_setup(configpath=os.path.join(cwd,configpath))
    
    main(dstpath=dstpath, labels_df=labels_df, fname_schema=fname_schema, cut_vals=cut_vals)