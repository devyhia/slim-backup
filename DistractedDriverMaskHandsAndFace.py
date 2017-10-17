import pickle
from glob import glob
import numpy as np
from PIL import Image
import Shared
from tqdm import tqdm
import os


with open('/home/devyhia/LendingAHand/result.pickle') as f:
    hands = pickle.load(f)

with open('/home/devyhia/FaceDetection_CNN/result.pickle') as f:
    faces = pickle.load(f)

def mask_image(img_path, boxes):
    img = Image.open(img_path)

    if len(boxes) == 0:
        return img

    mask = np.zeros((1080, 1920, 3))
    # box = boxes[0] # there is only one box per image! (by construction)
    for box in boxes:
        x0 = int(np.floor(box[0]))
        y0 = int(np.floor(box[1]))
        x1 = int(np.ceil(box[2]))
        y1 = int(np.ceil(box[3]))

        mask[y0:y1, x0:x1, :] = 1

    return Image.fromarray(mask.astype(np.uint8) * img) #  <-- Masked Image

img_count = 0
for k in tqdm(faces.keys(), total=len(faces.keys()), desc="Masking Hands & Face"):
    img_path = k.replace('\n', '') # avoid the \n at the end of each file!
    if '.original.jpg' not in img_path:
        img_path = img_path.replace('.jpg', '.original.jpg')

    faceBoxes = faces[k]

    pref = '.'.join(k.split('.')[:2])
    wOrig = pref + '.original.jpg\n'
    noOrig = pref + '.jpg\n'

    handBoxes = hands[wOrig] if wOrig in hands else hands[noOrig]

    boxes = faceBoxes + handBoxes

    save_path = img_path.replace('.original.jpg', '.hands_and_face.jpg')

    # if os.path.isfile(save_path): continue

    masked_img = mask_image(img_path, boxes)
    masked_img.save(save_path)
