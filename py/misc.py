from matplotlib import cm
from scipy import misc
import numpy as np
import matplotlib.pyplot as plt
from skimage import color
from skimage import data
from skimage import feature
from skimage import io

# im = misc.imread('../images/wheat.jpg', flatten=True)

im = misc.imread('../images/wheat.jpg')

r, g, b = np.rollaxis(im, -1)
plt.contour(b, cmap=cm.gray)
plt.contour(r, cmap=plt.cm.Reds)
plt.contour(g, cmap=plt.cm.Greens)
plt.contour(g, cmap=plt.cm.Blues)
plt.show()

# img_gray = color.convert_colorspace(im, 'RGB', 'Gray')

img_red = im[:, :, 0]
plt.imshow(img_red, cmap=plt.cm.Reds)
plt.show()

img_green = im[:, :, 1]
plt.imshow(img_green, cmap=plt.cm.Greens)
plt.show()

img_blue = im[:, :, 2]
plt.imshow(img_blue, cmap=plt.cm.Blues)
plt.show()

img_gray = color.rgb2gray(im)
plt.imshow(img_gray, cmap='gray')
plt.show()

img_hsv = color.rgb2hsv(im)
plt.imshow(img_hsv)
plt.show()

img_lab = color.rgb2lab(im)
plt.imshow(img_lab)
plt.show()

img_luv = color.rgb2luv(im)
plt.imshow(img_luv)
plt.show()

img_hed = color.rgb2hed(im)
plt.imshow(img_hed)
plt.show()
