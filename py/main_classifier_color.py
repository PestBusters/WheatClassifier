
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from numpy import random
from scipy import misc
from skimage import color
from skimage import filters
from skimage import exposure
from skimage import io
from sklearn.externals import joblib
from sklearn import cross_validation
from sklearn import grid_search
from sklearn.ensemble import RandomForestClassifier


clf = joblib.load('../data/classifier_color.clf')


def features_extractor(trimmed):

    from color_quantinization import quantinization
    image_gray = quantinization(trimmed)
    # image_gray = color.rgb2hsv(image_gray)

    featuresh = []

    hh = exposure.histogram(image_gray[:, :, 0])
    featuresh.extend(hh[0])

    hs = exposure.histogram(image_gray[:, :, 1])
    featuresh.extend(hs[0])

    hv = exposure.histogram(image_gray[:, :, 2])
    featuresh.extend(hv[0])

    return featuresh


# TRAIN

# files = io.ImageCollection('../images/examples/pure/' + '*.JPG')
files = io.ImageCollection('../images/examples/15p/' + '*.JPG')
for f in files.files:
    train_good_im = misc.imread(f)
    # train_good_img_gray = color.rgb2gray(train_good_im)
    train_good_img_gray = train_good_im

    for i in range(1, 2, 1):
        y = random.randint(0, train_good_img_gray.shape[0] - 101)
        x = random.randint(0, train_good_img_gray.shape[1] - 101)
        part = train_good_img_gray[y:y + 100, x:x + 100]
        print(clf.predict(features_extractor(part)))

