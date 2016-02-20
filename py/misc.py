from matplotlib import cm
from scipy import misc
import numpy as np
import matplotlib.pyplot as plt
from skimage import data
from skimage import feature
from skimage import io

# im = misc.imread('../images/wheat.jpg', flatten=True)
im = misc.imread('../images/wheat.jpg')

r, g, b = np.rollaxis(im, -1)
# plt.contour(b, cmap=cm.gray)
plt.contour(r, cmap=plt.cm.Reds)
plt.contour(g, cmap=plt.cm.Greens)
plt.contour(g, cmap=plt.cm.Blues)
plt.show()