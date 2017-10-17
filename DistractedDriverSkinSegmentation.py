import tensorflow as tf
import numpy as np
import os
from scipy.ndimage import imread
from scipy.misc import imsave
import matplotlib.colors as cl
import Shared
from tqdm import tqdm
import numpy as np
from sklearn import metrics
import tensorflow.contrib.distributions as tf_dist
import DistractedDriver as dd
from ipywidgets import widgets
from scipy.ndimage.measurements import label, labeled_comprehension
from glob import glob

Shared.select_gpu(3)

# print("Load dataset ...")
# _X, _y, _Xt, _yt = dd.load_data(True)

data = np.loadtxt("Skin_NonSkin.txt", delimiter="\t", unpack=False)
print("Loaded Skin/NonSkin Data from textfile ...")

Skin = list(filter(lambda e: e[3]==1, data))
NonSkin = list(filter(lambda e: e[3]==2, data))
print("Filtered Data into Skin/NonSkin ...")

X_Skin = np.array(Skin)
X_NonSkin = np.array(NonSkin)

trSkin = int(round(len(Skin)*1.0))
trNonSkin = int(round(len(NonSkin)*1.0))

Train_Skin = Skin[:trSkin]
Test_Skin = Skin[trSkin:]
Train_NonSkin = NonSkin[:trNonSkin]
Test_NonSkin = NonSkin[trNonSkin:]

print("Model Construction ...")
X = tf.placeholder(tf.float32, shape=(None, 3), name="X")
y = tf.placeholder(tf.float32, shape=(None, 1), name="y")

def ClassDistribution(scope):
    with tf.name_scope(scope):
        X = tf.placeholder(tf.float32, shape=(None, 3), name="X")
        sample_size = tf.cast(tf.shape(X)[0], dtype=tf.float32, name="sample_size")
        Mu = tf.reduce_mean(X, 0, name="Mu")
        X_norm = tf.sub(X, Mu, name="X_norm")
        Sigma = tf.div(tf.matmul(tf.transpose(X_norm), X_norm, name="Sigma"), sample_size)
        Dist = tf_dist.MultivariateNormalFull(Mu, Sigma, name="Dist")

        return {
            'X': X,
            'SampleSize': sample_size,
            'Mu': Mu,
            'Sigma': Sigma,
            'X_Normalized': X_norm,
            'Dist': Dist
        }

Skin = ClassDistribution("Skin")
NonSkin = ClassDistribution("NonSkin")

Pdf_Skin = Skin['Dist'].pdf(X)
Pdf_NonSkin = NonSkin['Dist'].pdf(X)
Tot_Pdf = Pdf_Skin + Pdf_NonSkin

Prob_Skin = Pdf_Skin / Tot_Pdf
Prob_NonSkin = Pdf_NonSkin / Tot_Pdf

Pred = Prob_NonSkin # as Label(NonSkin) = 1, Label(Skin) = 0
correct_prediction = tf.equal(tf.round(Pred), y)
Accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

print("Model Training ...")
with tf.Session() as sess:
    Skin_Mu, Skin_Sigma, NonSkin_Mu, NonSkin_Sigma = sess.run([
            Skin['Mu'], Skin['Sigma'], NonSkin['Mu'], NonSkin['Sigma']
        ], feed_dict={
            Skin['X']: cl.rgb_to_hsv(X_Skin[:, [2,1,0]]),
            NonSkin['X']: cl.rgb_to_hsv(X_NonSkin[:, [2,1,0]])
        })

def process_image(fltr, img, threshold):
    lbl, nlbl = label(fltr > threshold)
    lbls = np.arange(1, nlbl+1)
    objs = labeled_comprehension(fltr, lbl, lbls, np.count_nonzero, int, 0)
    main_objs = np.arange(0, objs.shape[0])[objs>1]
    score = main_objs.sum()
    top_3_lbls = reduce(lambda prev, curr: prev + (lbl == curr+1).astype(int), main_objs, np.zeros(lbl.shape))
    img_w_lbls = np.where(top_3_lbls[:,:, np.newaxis].astype(np.bool), img, np.zeros(img.shape)).astype(np.uint8)
    return lbl, nlbl, objs, main_objs, score, top_3_lbls, img_w_lbls

def segment_images(images, batch_size=20, threshold=0.5):
    fltrs = np.zeros(images.shape[:3])
    top_3_fltrs = np.zeros(images.shape[:3])
    images_segmented = np.zeros(images.shape)

    count = images.shape[0]
    batches = range(0, count, batch_size) + [count]
    with tf.Session() as sess:
        pbar = tqdm(total=count)
        for start, end in zip(batches[:-1], batches[1:]):
            fltrs[start:end] = sess.run(Prob_Skin, feed_dict={
                    X: cl.rgb_to_hsv(images[start:end]).reshape((-1, 3)),
                    Skin['Mu']: Skin_Mu,
                    Skin['Sigma']: Skin_Sigma,
                    NonSkin['Mu']: NonSkin_Mu,
                    NonSkin['Sigma']: NonSkin_Sigma
                }).reshape((-1, 299, 299))

            for i in range(start, end):
                lbl, nlbl, objs, main_objs, score, top_3_lbls, img_w_lbls = process_image(
                    fltrs[i], images[i], threshold
                )
                top_3_fltrs[i] = top_3_lbls
                images_segmented[i] = img_w_lbls
                pbar.update(1)

    return fltrs, top_3_fltrs, images_segmented

def in_place_images_segmentation(threshold=0.5):
    images = glob('/home/devyhia/distracted.driver/*/*.original.jpg')

    raw_shape = (1080, 1920)
    fltrs = np.zeros(raw_shape)

    with tf.Session() as sess:
        pbar = tqdm(total=len(images))
        for i, img_path in enumerate(images):
            if i < 1239:
                pbar.update(1)
                continue
                
            img = imread(img_path)
            fltrs = sess.run(Prob_Skin, feed_dict={
                    X: cl.rgb_to_hsv(img).reshape((-1, 3)),
                    Skin['Mu']: Skin_Mu,
                    Skin['Sigma']: Skin_Sigma,
                    NonSkin['Mu']: NonSkin_Mu,
                    NonSkin['Sigma']: NonSkin_Sigma
                }).reshape(raw_shape)

            lbl, nlbl, objs, main_objs, score, top_3_lbls, img_w_lbls = process_image(
                fltrs, img, threshold
            )
            imsave(img_path.replace('.original.', '.segmented.'), img_w_lbls)
            pbar.update(1)


# print("Segmenting training images ...")
# fltrs, fltrs_smoothed, images_segmented = segment_images(_X, batch_size=50)
#
# print("Saving segmented training images ...")
# np.save("cache/X.segmented.npy", images_segmented.astype(np.uint8))
#
# print("Segmenting testing images ...")
# fltrs, fltrs_smoothed, images_segmented = segment_images(_Xt, batch_size=50)
#
# print("Saving segmented testing images ...")
# np.save("cache/Xt.segmented.npy", images_segmented.astype(np.uint8))

print("Segmenting the dataset in original size ...")
in_place_images_segmentation() # Start the segmentation!
