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

image_gray = color.rgb2gray(image)
# image_gray = color.rgb2hsv(image)[:, :, 1]  # HUE
# plt.imshow(image_gray, cmap='gray')
# plt.show()


# image = img_as_ubyte(data.coins()[0:95, 70:370])
image = img_as_ubyte(image_gray)[0:200, 0:200]
# image = img_as_ubyte(data.coins()[0:95, 70:370])
edges = canny(image, sigma=3, low_threshold=10, high_threshold=50)
plt.imshow(edges)
plt.show()


fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(5, 2))

# Detect two radii
hough_radii = np.arange(15, 30, 2)
# hough_radii = np.arange(10, 10, 2)
# hough_radii = np.array([10, 20, 30, 40])
hough_radii = np.arange(20, 40, 1)
hough_res = hough_circle(edges, hough_radii)

centers = []
accums = []
radii = []

for radius, h in zip(hough_radii, hough_res):
    # For each radius, extract two circles
    num_peaks = 2
    peaks = peak_local_max(h, num_peaks=num_peaks)
    centers.extend(peaks)
    accums.extend(h[peaks[:, 0], peaks[:, 1]])
    radii.extend([radius] * num_peaks)

# Draw the most prominent 5 circles
image = color.gray2rgb(image)
for idx in np.argsort(accums)[::-1][:5]:
    center_x, center_y = centers[idx]
    radius = radii[idx]
    cx, cy = circle_perimeter(center_y, center_x, radius)
    image[cy, cx] = (220, 20, 20)

ax.imshow(image, cmap=plt.cm.gray)

plt.show()