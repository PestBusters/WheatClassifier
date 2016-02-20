import matplotlib.pyplot as plt

from skimage import data, color
from skimage.feature import canny
from skimage.transform import hough_ellipse
from skimage.draw import ellipse_perimeter
import numpy as np
# from numpy import arange
import matplotlib.pyplot as plt
from skimage.morphology import disk
from scipy import misc
from skimage import color
from skimage import measure
from skimage import io

# Load some test data
# image = misc.imread('../images/wheat3.jpg')
from skimage.filters.rank import autolevel

files = io.ImageCollection('../images/examples/pure/' + '*.JPG')
im = misc.imread(files.files[0])

image_gray = color.rgb2gray(im)[0:100, 0:100]
# image_gray = color.rgb2gray(im)[0:200, 0:200]
# image_gray = color.rgb2hsv(im)[:, :, 2]  # HUE
plt.imshow(image_gray, cmap='gray')
plt.show()

# Load picture, convert to grayscale and detect edges
# image_rgb = data.coffee()[0:220, 160:420]
image_rgb = im
# image_gray = color.rgb2gray(image_rgb)
image_gray = image_gray
# edges = canny(image_gray, sigma=2.0, low_threshold=0.55, high_threshold=0.8)
edges = canny(image_gray, sigma=0.1)
# edges = canny(image_gray)
plt.imshow(edges)
plt.show()
# Perform a Hough Transform
# The accuracy corresponds to the bin size of a major axis.
# The value is chosen in order to get a single high accumulator.
# The threshold eliminates low accumulators
# result = hough_ellipse(edges, accuracy=20, threshold=250, min_size=10, max_size=20)
result = hough_ellipse(edges, threshold=250, min_size=10, max_size=20)
result.sort(order='accumulator')

# Estimated parameters for the ellipse
best = list(result[-1])
yc, xc, a, b = [int(round(x)) for x in best[1:5]]
orientation = best[5]

# Draw the ellipse on the original image
cy, cx = ellipse_perimeter(yc, xc, a, b, orientation)
image_rgb[cy, cx] = (0, 0, 255)
# Draw the edge (white) and the resulting ellipse (red)
edges = color.gray2rgb(edges)
edges[cy, cx] = (250, 0, 0)

fig2, (ax1, ax2) = plt.subplots(ncols=2, nrows=1, figsize=(8, 4), sharex=True,
                                sharey=True,
                                subplot_kw={'adjustable':'box-forced'})

ax1.set_title('Original picture')
ax1.imshow(image_rgb)

ax2.set_title('Edge (white) and result (red)')
ax2.imshow(edges)

plt.show()