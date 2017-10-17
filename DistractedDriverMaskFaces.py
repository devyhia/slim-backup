import pickle
from glob import glob
import numpy as np
from PIL import Image
import Shared
from tqdm import tqdm
import os


with open('/home/devyhia/FaceDetection_CNN/result.pickle') as f:
    results = pickle.load(f)

def mask_image(img_path, boxes):
    img = Image.open(img_path)

    if len(boxes) == 0:
        return img

    box = boxes[0] # there is only one box per image! (by construction)

    x0 = int(np.floor(box[0]))
    y0 = int(np.floor(box[1]))
    x1 = int(np.ceil(box[2]))
    y1 = int(np.ceil(box[3]))

    if x0 > 0.5 * 1920: # Face can not be in the right corner of the image (by construction)
        return img

    mask = np.zeros((1080, 1920, 3))
    mask[y0:y1, x0:x1, :] = 1

    return Image.fromarray(mask.astype(np.uint8) * img) #  <-- Masked Image

img_count = 0
for k in tqdm(results.keys(), total=len(results.keys()), desc="Masking Faces"):
    img_path = k.replace('\n', '').replace('.jpg', '.original.jpg') # avoid the \n at the end of each file!
    boxes = results[k]

    save_path = img_path.replace('.original.jpg', '.face.jpg')

    # if os.path.isfile(save_path): continue

    masked_img = mask_image(img_path, boxes)
    masked_img.save(save_path)
