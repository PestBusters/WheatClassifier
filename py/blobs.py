import numpy as np
import matplotlib.pyplot as plt
from math import sqrt
from scipy import misc
from skimage import io
from skimage import color
from skimage import feature
from skimage import restoration
from skimage import filter

# image = misc.imread('../images/wheat3.jpg')
# image = files = io.ImageCollection('../images/examples/pure' + '*.jpg')[0]
# image = io.ImageCollection('../images/examples/pure/' + '*.JPG')[1]
# image = io.ImageCollection('../images/examples/pure/' + 'SAM_0681.JPG')[0]
# image = misc.imread('../images/examples/pure/' + 'SAM_0681.JPG')
image = misc.imread('../images/examples/15p/' + 'SAM_0712.JPG')
plt.imshow(image)
plt.show()

# image_gray = color.rgb2gray(image)
image_gray = color.rgb2hsv(image)[:, :, 1]  # HUE
image_gray = image_gray[100:200, 100:200]

# filter_blurred_f = filter.gaussian_filter(image_gray, 1)
# alpha = 0.1
# sharpened = image_gray + alpha * (image_gray - filter_blurred_f)
# image_gray = sharpened

edge_sobel = filter.sobel(image_gray)
image_gray = edge_sobel
# psf = np.ones((5, 5)) / 25
# image_gray, _ = restoration.unsupervised_wiener(image_gray, psf)

plt.imshow(image_gray, cmap='gray')
plt.show()

# blobs_log = feature.blob_log(image_gray, max_sigma=30, num_sigma=10, threshold=.1)
# plt.imshow(blobs_log, cmap='gray')
# plt.show()

blobs_log = feature.blob_log(image_gray, max_sigma=30, num_sigma=10, threshold=.1)

# Compute radii in the 3rd column.
blobs_log[:, 2] = blobs_log[:, 2] * sqrt(2)

blobs_dog = feature.blob_dog(image_gray, max_sigma=30, threshold=.1)
blobs_dog[:, 2] = blobs_dog[:, 2] * sqrt(2)

blobs_doh = feature.blob_doh(image_gray, max_sigma=30, threshold=.01)

blobs_list = [blobs_log, blobs_dog, blobs_doh]
colors = ['yellow', 'lime', 'red']
titles = ['Laplacian of Gaussian', 'Difference of Gaussian', 'Determinant of Hessian']
sequence = zip(blobs_list, colors, titles)

fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharex=True, sharey=True,
                         subplot_kw={'adjustable': 'box-forced'})
plt.tight_layout()

axes = axes.ravel()
for blobs, color, title in sequence:
    ax = axes[0]
    axes = axes[1:]
    ax.set_title(title)
    ax.imshow(image, interpolation='nearest')
    ax.set_axis_off()
    for blob in blobs:
        y, x, r = blob
        c = plt.Circle((x, y), r, color=color, linewidth=2, fill=False)
        ax.add_patch(c)

plt.show()
