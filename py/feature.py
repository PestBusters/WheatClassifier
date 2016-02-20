from skimage import data, feature

import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction import image
from math import sqrt
from scipy import misc
from skimage import io
from skimage import color
from skimage import feature
from skimage import restoration
from skimage import filter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.externals import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn import neighbors, svm, metrics, cluster
from itertools import count
from os.path import exists
import pylab as pl
from datetime import datetime
import csv
import sys


# files = io.ImageCollection('../images/examples/pure/' + '*.JPG')
# for file in files.files:

train_good_im = misc.imread('../images/examples/pure/SAM_0681.JPG')
train_good_img_gray = color.rgb2gray(train_good_im)[0:300, 0:300]
# train_good_patches = image.extract_patches_2d(train_good_img_gray, (2, 2), max_patches=0.3, random_state=0)
train_good_mean = np.mean(train_good_img_gray)
train_good_sum = np.sum(train_good_img_gray)
train_good_avg = np.average(train_good_img_gray)
# print(patches)

train_bad_im = misc.imread('../images/examples/15p/SAM_0712.JPG')
train_bad_img_gray = color.rgb2gray(train_bad_im)[0:300, 0:300]
train_bad_patches = image.extract_patches_2d(train_bad_img_gray, (2, 2), max_patches=0.3, random_state=0)

train_bad_mean = np.mean(train_bad_img_gray)
train_bad_sum = np.sum(train_bad_img_gray)
train_bad_avg = np.average(train_bad_img_gray)


#  TEST

test_good_im = misc.imread('../images/examples/pure/SAM_0681.JPG')
test_good_img_gray = color.rgb2gray(train_good_im)[100:400, 100:400]
test_good_patches = image.extract_patches_2d(train_good_img_gray, (2, 2), max_patches=0.3, random_state=0)
# print(patches)

test_good_mean = np.mean(test_good_img_gray)
test_good_sum = np.sum(test_good_img_gray)
test_good_avg = np.average(test_good_img_gray)


test_bad_im = misc.imread('../images/examples/15p/SAM_0712.JPG')
test_bad_img_gray = color.rgb2gray(train_bad_im)[100:400, 100:400]
test_bad_patches = image.extract_patches_2d(train_bad_img_gray, (2, 2), max_patches=0.3, random_state=0)

test_bad_mean = np.mean(test_bad_img_gray)
test_bad_sum = np.sum(test_bad_img_gray)
test_bad_avg = np.average(test_bad_img_gray)

train = [
    [train_good_mean, train_good_sum, train_good_avg],
    [train_bad_mean, train_bad_sum, train_bad_avg]
]

target = [
    1,
    0
]

test = [
    [test_good_mean, test_good_sum, test_good_avg],
    [test_bad_mean, test_bad_sum, test_bad_avg]
]

import operator

# train = []
# target = []
# for f in train_bad_patches:
#     # print f
#     z = reduce(operator.add, list(f))
#     # print z
#     # print z.shape
#     # train[f] = 0
#     train.append(f)
#     target.append(0)
#
# for f in train_good_patches:
#     # train[f] = 1
#     z = reduce(operator.add, list(f))
#     train.append(z)
#     target.append(1)

classifier = RandomForestClassifier(n_estimators=127, n_jobs=-1, verbose=1)
# classifier.fit(train.keys(), train.values())
# print(train.shape)
classifier.fit(train, target)

# for f in test_bad_patches:
#     z = reduce(operator.add, list(f))
    # prediction = [float(x[-1]) for test_bad_patches in classifier.predict()]
prediction = classifier.predict(test[1])
print('expected 0 predicted {}'.format(prediction))

# for f in test_good_patches:
#     z = reduce(operator.add, list(f))
prediction = classifier.predict(test[0])
print('expected 1 predicted {}'.format(prediction))
