import numpy as np
# from numpy import arange
import matplotlib.pyplot as plt
from scipy import misc
from skimage import color
from skimage import measure

# Load some test data
image = misc.imread('../images/wheat3.jpg')

image_gray = color.rgb2gray(image)
# image_gray = color.rgb2hsv(image)[:, :, 1]  # HUE
# plt.imshow(image_gray, cmap='gray')
# plt.show()

for x in np.arange(0.1, 1.0, 0.1):
    # Find contours at a constant value of 0.8
    contours = measure.find_contours(image_gray, x)

    # Display the image and plot all contours found
    fig, ax = plt.subplots()
    ax.imshow(image_gray, interpolation='nearest', cmap=plt.cm.gray)

    for n, contour in enumerate(contours):
        ax.plot(contour[:, 1], contour[:, 0], linewidth=2)

    ax.axis('image')
    ax.set_xticks([])
    ax.set_yticks([])
    plt.show()