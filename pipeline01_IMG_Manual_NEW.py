import cv2
import numpy as np
import pandas as pd
import scipy
import matplotlib.pyplot as plt
plt.rcParams['image.cmap'] = 'gray'
from matplotlib import transforms
import os
import inspect
import yaml
from sklearn.metrics import confusion_matrix

best_val = 0.0

def predict(img, fpath, smoothig_val, moving_av, img_orig, peakthr=2, make_plot=False):
    # analysing the image
    df = pd.DataFrame(img)
    colsum=df.sum(axis=1) #adding up columns to have the distribution of white pixels over the rows
    max=colsum.max() #  determining maximum value for threshold later on for noise reduction
    # some smoothing has to be applied to filter out noise
    colsum = pd.Series(colsum.rolling(moving_av).mean())
    colsum = colsum.apply(lambda x: 0 if x <= max/100*smoothig_val or np.isnan(x) else x)
    #finding peaks
    peaks, props = scipy.signal.find_peaks(colsum, height=0)

    #predicting
    predictfunc = lambda peaks: "defect" if len(peaks)>peakthr else "nodefect"
    funcString = str(inspect.getsourcelines(predictfunc)[0])
    funcString = funcString.strip("['\\n']").split(" = ")[1]
    prediction = predictfunc(peaks)

    pred = {"fname": os.path.split(fpath)[1].split('.')[0],
            "peaks": len(peaks),
            "prediction": prediction,
            "criteria": funcString}

    if make_plot:
        plot_overview(img_orig, img, colsum, peaks, fpath)
    return pred


def plot_overview(img_orig, img_thr, colsum, peaks, fpath):
    path, fname = os.path.split(fpath)
    fname = fname.split('.')[0]
    rootpath = os.path.split(path)[0]
    savepath = os.path.join(rootpath, 'overview',fname)

    aspectratio = "auto"
    fig, axs = plt.subplots(1,4,figsize=(10,5), sharey=True)
    bases=[]
    for i in range(len (axs)):
        bases.append(axs[i].transData)
    rot = transforms.Affine2D().rotate_deg(90)
    axs[0].imshow(img_orig, aspect = aspectratio)
    axs[0].title.set_text(f"Original")
    axs[1].imshow(img_thr, aspect = aspectratio)
    axs[1].title.set_text(f"Processed")
    axs[2].plot(colsum,transform= rot + bases[2])
    axs[2].title.set_text(f"Pixel Distribution smoothed")
    axs[2].invert_xaxis()
    axs[3].plot(colsum,transform= rot + bases[3])
    axs[3].plot(peaks, colsum[peaks], "x", transform= rot + bases[3])
    axs[3].title.set_text(f"Local Maxima = {len(peaks)}")
    axs[3].invert_xaxis()
    for ax in axs:
        ax.tick_params(labelrotation=45)
    # plt.show()
    fig.savefig(savepath)
    plt.close('all')

def thr_img(img, thrval):
    _, threshold = cv2.threshold(img, thrval, 255, cv2.THRESH_BINARY)
    return threshold

def ero_dil_img(img, ero_iters, dil_iters):
    kernel = np.array(
                        [[0.,0.,0.],
                        [1.,1.,1.],
                        [0.,0.,0.]],np.uint8)
    img_erosion = cv2.erode(img, kernel, iterations=ero_iters)
    img_dilation = cv2.dilate(img_erosion, kernel, iterations=dil_iters)
    return img_dilation

def load_new_image(imgpath):
    img = cv2.imread(imgpath)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img

def get_cm(cm):
    class_names = ["nodefect", 'defect']
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.matshow(cm, cmap=plt.cm.Blues, alpha=0.3)
    for i in range(len(cm)):
        for j in range(len(cm[1])):
            ax.text(x=j, y=i,s=cm[i][j], va='center', ha='center', size='xx-large')
    plt.xlabel('Predictions', fontsize=18)
    ax.xaxis.tick_bottom()
    ax.set_xticks([0,1])
    ax.set_xticklabels(class_names)
    plt.ylabel('Actuals', fontsize=18)
    ax.set_yticks([0,1])
    ax.set_yticklabels(class_names)
    plt.title(f'Confusion Matrix', fontsize=18)
    return fig

def evaluate(y_true, y_pred, dstpath, trialnr):
    global best_val
    is_pred_right= list(y_true == y_pred)
    accuracy = is_pred_right.count(True)/len(y_pred)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    precision=tp/(tp+fp) if fp !=0 and tp != 0 else 1
    recall=tp/(tp+fn)
    f1=tp/(tp+(fn+fp)/2)

    full_eval = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
    }


    if accuracy >= best_val:
        best_val = accuracy
        fig = get_cm(cm)
        fig.savefig(os.path.join(dstpath,"cm",f'trial{trialnr}'))
        
    return accuracy

def run_dataset(dataset, config):

    thrval = config["Processing"]["thrval"]
    ero_iters = config["Processing"]["ero_iters"]
    dil_iters = config["Processing"]["dil_iters"]
    smoothing_val = config["Processing"]["smoothing_val"]
    movinge_av = config["Processing"]["movinge_av"]
    #setting up environment variables
    make_plot = True
    predictions = pd.DataFrame(columns=["fname","peaks","prediction", "criteria"])
    
    for path in list(dataset.paths):
        img = load_new_image(path)
        img_thr = thr_img(img, thrval)
        img_erodil = ero_dil_img(img_thr, ero_iters, dil_iters)
        preds = predict(img_erodil, path, smoothing_val, movinge_av, img, make_plot=make_plot)
        predictions.loc[len(predictions)] = preds
        make_plot = False
    return predictions

def main(trial=None):
    #read variables from config
    with open('configs/pipeline01config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    dstpath = config["Paths"]["dstpath"]
    labelsfname = config["Paths"]["labelsfname"]
    if trial==None:
        trialnr=0
    else:
        trialnr = trial.number

    #loading and splitting dataset
    df_labels = pd.read_csv(os.path.join(dstpath,labelsfname),index_col=0)
    
    y_true = df_labels[["paths","labels"]]
    y_true.paths = y_true['paths'].apply(lambda x: os.path.split(x)[1].split('.')[0])
    y_true.columns = ["fname","label"]
    y_true.label = y_true['label'].apply(lambda x: {'defect': 1, 'nodefect': 0}[x])
    y_pred = run_dataset(df_labels, config)
    y_pred = y_pred[["fname","prediction"]]
    y_pred.prediction = y_pred['prediction'].apply(lambda x: {'defect': 1, 'nodefect': 0}[x])
    df_eval = y_true.join(y_pred.set_index('fname'), on='fname')
    acc = evaluate(df_eval["label"], df_eval['prediction'], dstpath, trialnr)
    return acc
if __name__ == '__main__':
    main()