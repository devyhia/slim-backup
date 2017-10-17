import urllib
import pandas as pd
from glob import glob
from helpers import update_screen

df = pd.read_csv('/home/devyhia/annotations/no_persons.txt', header=None)
N = df.shape[0]

for i in range(N):
    image_id = df.iloc[i, 0]
    save_to = "/home/devyhia/coco-no-persons/{}.jpg".format(image_id)
    download_url = "http://mscoco.org/images/{}".format(image_id)

    if glob(save_to): # if file is found (i.e. len > 0)
        continue

    urllib.urlretrieve (download_url, save_to)
    update_screen('\r{} out of {}'.format(i+1, N))
