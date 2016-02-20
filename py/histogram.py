import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin
from sklearn.datasets import load_sample_image
from sklearn.utils import shuffle
from scipy import misc
from skimage import color
from skimage import measure
from skimage import exposure
from skimage import io
from time import time

files = io.ImageCollection('../images/examples/pure/' + '*.JPG')
# files = io.ImageCollection('../images/examples/15p/' + '*.JPG')
im = misc.imread(files.files[0])

# image_gray = color.rgb2gray(im)
# image_gray = color.rgb2hsv(im)[:, :, 0]
# image_gray = color.rgb2hsv(im)
# plt.imshow(image_gray)
# plt.show()

from color_quantinization import quantinization
image_gray = quantinization(im)
# image_gray = im
# image_gray = color.rgb2hsv(image_gray)[:, :, 0]
image_gray = color.rgb2hsv(image_gray)
# h = exposure.histogram(image_gray[:, :, 0])
# h = exposure.histogram(image_gray[:, :, 1])
# h = exposure.histogram(image_gray[:, :, 2])
h = exposure.histogram(image_gray[:, :, 2])

# print(h[0].shape)
# print(h[0])

plt.hist(h[0], color='gray')
# plt.hist(h[1], color='green')
# plt.hist(h[2], color='black')
plt.show()
