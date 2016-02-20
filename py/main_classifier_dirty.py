
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


clf = joblib.load('../data/classifier_dirty.clf')


def features_extractor(trimmed):
    return [
        np.median(trimmed),
        np.sum(trimmed),
        np.average(trimmed),
        np.max(trimmed),
        np.min(trimmed)
    ]


# files = io.ImageCollection('../images/examples/pure/' + '*.JPG')
files = io.ImageCollection('../images/examples/10p/' + '*.JPG')
for f in files.files:
    train_good_im = misc.imread(f)
    train_good_img_gray = color.rgb2gray(train_good_im)

    for i in range(1, 10, 1):
        y = random.randint(0, train_good_img_gray.shape[0] - 101)
        x = random.randint(0, train_good_img_gray.shape[1] - 101)
        part = train_good_img_gray[y:y + 100, x:x + 100]
        print(clf.predict(features_extractor(part)))

