from PIL import Image
import os
import numpy as np

def get_folder_RGB(folder):
    total_images=[]
    filenames=os.listdir(folder)
    for f in filenames:
        path=os.path.join(folder,f)
        img=Image.open(path)
        img=img.convert("RGB")
        img=img.resize((128,128))
        arr=np.array(img)
        total_images.append(arr)
    return(total_images)    
