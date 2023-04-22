import glob
import shutil
import os
import yaml

def get_imagepaths(fending=".png", rootpath = None): # loads source images
    image_list = glob.glob(f"{rootpath}/*/*{fending}", recursive=True)
    if image_list == []:
        image_list = glob.glob(f"{rootpath}/*{fending}", recursive=True)
    image_list.sort()
    raw_image_list = image_list
    return image_list

def clean_dir(path):
    try:
        shutil.rmtree(path)
    except: pass
    os.makedirs(path)

def load_setup(configpath): # loads config from config.yaml
    with open(configpath, 'r') as file:
        config = yaml.safe_load(file)
    return config

def gen_filename(path2folder, filename):
    i = 1
    try:
        while os.path.exists(f"{path2folder}/{filename}{i:03d}.png"):
            i += 1
    except: pass
    return f"{path2folder}/{filename}{i:03d}.png"
