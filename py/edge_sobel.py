import numpy as np
import matplotlib.pyplot as plt

from scipy import misc
from skimage import color
from skimage import measure
from skimage import io
from skimage.data import camera
from skimage.filters import roberts, sobel, scharr, prewitt

files = io.ImageCollection('../images/examples/pure/' + '*.JPG')
im = misc.imread(files.files[0])

# image_gray = color.rgb2gray(im)[0:100, 0:100]
image_gray = color.rgb2hsv(im)[:, :, 2][0:200, 0:200]

# image = camera()
edge_roberts = roberts(image_gray)
edge_sobel = sobel(image_gray)

fig, (ax0, ax1) = plt.subplots(ncols=2, sharex=True, sharey=True, subplot_kw={'adjustable':'box-forced'})

ax0.imshow(edge_roberts, cmap=plt.cm.gray)
ax0.set_title('Roberts Edge Detection')
ax0.axis('off')

ax1.imshow(edge_sobel, cmap=plt.cm.gray)
ax1.set_title('Sobel Edge Detection')
ax1.axis('off')

plt.tight_layout()
plt.show()
