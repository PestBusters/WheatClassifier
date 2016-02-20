from math import sqrt

import matplotlib.pyplot as plt
from scipy import misc
from skimage import color
from skimage import feature

image = misc.imread('../images/wheat3.jpg')

image_gray = color.rgb2gray(image)
# image_gray = color.rgb2hsv(image)[:, :, 1]  # HUE
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
