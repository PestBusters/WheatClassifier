import numpy as np
import matplotlib.pyplot as plt

from skimage import data, color
from skimage.transform import hough_circle
from skimage.feature import peak_local_max, canny
from skimage.draw import circle_perimeter
from skimage.util import img_as_ubyte
from scipy import misc
from skimage import color
from skimage import measure


# Load picture and detect edges

# Load some test data
image = misc.imread('../images/wheat3.jpg')

image = image[0:200, 0:200]
plt.imshow(image)
plt.show()

image_gray = color.rgb2gray(image)
# image_gray = color.rgb2hsv(image)[:, :, 1]  # HUE
# plt.imshow(image_gray, cmap='gray')
# plt.show()


# image = img_as_ubyte(data.coins()[0:95, 70:370])
image = img_as_ubyte(image_gray)
# image = image_gray

edges = canny(image, sigma=1, low_threshold=1, high_threshold=5)
plt.imshow(edges)
plt.show()
