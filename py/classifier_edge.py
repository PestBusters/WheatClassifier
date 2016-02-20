from datetime import datetime

import numpy as np
from numpy import random
from scipy import misc
from skimage import color
from skimage import filter
from skimage import io
from sklearn import cross_validation
from sklearn import grid_search
from sklearn.ensemble import RandomForestClassifier

train = []
train_imgs = []
train_f = []
target = []


def features_extractor(trimmed):
    # edge_roberts = filter.roberts(trimmed)
    edge_sobel = filter.sobel(trimmed)
    return [
        # np.median(trimmed),
        # np.sum(trimmed),
        # np.average(trimmed),
        # np.max(trimmed),
        # np.min(trimmed)
        # np.sum([x for x in edge_roberts.flatten() if x > 100]) / np.sum([x for x in edge_roberts.flatten() if x <= 100]),
        # np.sum([x for x in edge_sobel.flatten() if x > 100]) / np.sum([x for x in edge_sobel.flatten() if x <= 100])
        np.sum([x for x in edge_sobel.flatten() if x > 100]),
        len([x for x in edge_sobel.flatten() if x > 100])
    ]


files = io.ImageCollection('../images/examples/pure/' + '*.JPG')
for f in files.files:
    train_good_im = misc.imread(f)
    train_good_img_gray = color.rgb2gray(train_good_im)

    for i in range(1, 10, 1):
        y = random.randint(0, train_good_img_gray.shape[0] - 101)
        x = random.randint(0, train_good_img_gray.shape[1] - 101)
        part = train_good_img_gray[y:y + 100, x:x + 100]
        train.append(features_extractor(part))
        target.append(1)

files = []
files.extend(io.ImageCollection('../images/examples/15p/' + '*.JPG').files)
files.extend(io.ImageCollection('../images/examples/10p/' + '*.JPG').files)
files.extend(io.ImageCollection('../images/examples/5p/' + '*.JPG').files)
for f in files:
    train_bad_im = misc.imread(f)
    train_bad_img_gray = color.rgb2gray(train_bad_im)
    for i in range(1, 10, 1):
        y = random.randint(0, train_bad_img_gray.shape[0] - 101)
        x = random.randint(0, train_bad_img_gray.shape[1] - 101)
        part = train_bad_img_gray[y:y + 100, x:x + 100]
        train.append(features_extractor(part))
        target.append(0)

# TEST

test = []
test_imgs = []
expected = []

files = io.ImageCollection('../images/examples/pure/' + '*.JPG')
for f in files.files:
    train_good_im = misc.imread(f)
    train_good_img_gray = color.rgb2gray(train_good_im)
    for i in range(1, 10, 1):
        y = random.randint(0, train_good_img_gray.shape[0] - 101)
        x = random.randint(0, train_good_img_gray.shape[1] - 101)
        part = train_good_img_gray[y:y + 100, x:x + 100]
        test.append(features_extractor(part))
        expected.append(1)

files = []
files.extend(io.ImageCollection('../images/examples/15p/' + '*.JPG').files)
files.extend(io.ImageCollection('../images/examples/10p/' + '*.JPG').files)
files.extend(io.ImageCollection('../images/examples/5p/' + '*.JPG').files)
for f in files:
    train_bad_im = misc.imread(f)
    train_bad_img_gray = color.rgb2gray(train_bad_im)
    for i in range(1, 10, 1):
        y = random.randint(0, train_bad_img_gray.shape[0] - 101)
        x = random.randint(0, train_bad_img_gray.shape[1] - 101)
        part = train_bad_img_gray[y:y + 100, x:x + 100]
        test.append(features_extractor(part))
        expected.append(0)

train = [np.nan_to_num(x) for x in train]
test = [np.nan_to_num(x) for x in test]

ts = datetime.now()

parameters = {
    # 'n_estimators': range(10, 10000, 1000)
    'n_estimators': [4096]
}

# classifier = RandomForestClassifier(n_estimators=2048, n_jobs=-1, verbose=1)
classifier = RandomForestClassifier(n_jobs=-1, verbose=1)
# classifier.fit(train, target)

clf = grid_search.GridSearchCV(classifier, parameters, verbose=2)
clf.fit(train, target)

clf.verbose = 0
classifier.verbose = 0

print("best estimator %s " % clf.best_estimator_)
print("best params: %s " % clf.best_params_)
print("best score %s " % clf.best_score_)

n_samples = np.min([len(train), len(test)])
cv = cross_validation.ShuffleSplit(n_samples, n_iter=5, test_size=0.3, random_state=15)

scores = cross_validation.cross_val_score(clf.best_estimator_, train, target, cv=cv)
# scores = cross_validation.cross_val_score(clf, train, target, cv=clf.best_estimator_)
print("Accuracy train: %0.4f (+/- %0.4f)" % (scores.mean(), scores.std() * 2))

scores = cross_validation.cross_val_score(clf.best_estimator_, test, expected, cv=cv)
# scores = cross_validation.cross_val_score(clf, test, expected, cv=clf.best_estimator_)
print("Accuracy test : %0.4f (+/- %0.4f)" % (scores.mean(), scores.std() * 2))

# for i in range(0,len(test)):
#     prediction = classifier.predict(test[i])
#     print('expected {} actual {}'.format(expected[i], prediction))

te = datetime.now()
print('elapsed time: {0}'.format(te - ts))
