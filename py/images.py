import numpy as np
import matplotlib.pyplot as plt
from skimage import io
import os

# draw test image
# check = np.zeros((9, 9))
# check[::2, 1::2] = 1
# check[1::2, ::2] = 1
# plt.imshow(check, cmap='gray', interpolation='nearest')
# plt.show()

# draw single image by filename
image_dir = '../images/'
image_file = 'wheat.jpg'
file = os.path.join(image_dir, image_file)
image = io.imread(file)
# plt.imshow(image)
# plt.show()

# draw all files by pattern
# files = io.ImageCollection(image_dir + '*.jpg')
# for image in files:
#     plt.imshow(image)
#     plt.show()


# filters
from skimage.filters import roberts, sobel, scharr, prewitt
from scipy import misc
# im = misc.imread('../images/wheat.jpg', flatten=True)
im = misc.imread('../images/wheat.jpg')

# edge_roberts = roberts(image[:, :, 0])
edge_roberts = roberts(im[:, :, 0])
# edge_roberts = roberts(im)
# edge_sobel = sobel(image[:, :, 0])
edge_sobel = sobel(im[:, :, 0])
# edge_sobel = sobel(im)


fig, (ax0, ax1) = plt.subplots(ncols=2, sharex=True, sharey=True, subplot_kw={'adjustable':'box-forced'})

ax0.imshow(edge_roberts, cmap=plt.cm.gray)
ax0.set_title('Roberts Edge Detection')
ax0.axis('off')

ax1.imshow(edge_sobel, cmap=plt.cm.gray)
ax1.set_title('Sobel Edge Detection')
ax1.axis('off')

# plt.tight_layout()
plt.show()