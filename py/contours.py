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

image_gray = color.rgb2gray(im)
# image_gray = color.rgb2hsv(im)[:, :, 2]  # HUE
# plt.imshow(image_gray, cmap='gray')
# plt.show()

# for x in np.arange(0.1, 1.0, 0.1):
for x in [0.5]:
    # Find contours at a constant value of 0.8
    # image_gray = autolevel(image_gray.astype(np.uint16), disk(1))
    contours = measure.find_contours(image_gray, x)

    # Display the image and plot all contours found
    fig, ax = plt.subplots()
    ax.imshow(image_gray, interpolation='nearest', cmap=plt.cm.gray)

    for n, contour in enumerate(contours):
        ax.plot(contour[:, 1], contour[:, 0], linewidth=1)

    ax.axis('image')
    ax.set_xticks([])
    ax.set_yticks([])
    plt.show()