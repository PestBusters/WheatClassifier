import numpy as np
import matplotlib.pyplot as plt
from skimage import io
import os

# draw test image
check = np.zeros((9, 9))
check[::2, 1::2] = 1
check[1::2, ::2] = 1
plt.imshow(check, cmap='gray', interpolation='nearest')
plt.show()

# draw single image by filename
image_dir = '../images/'
image_file = 'wheat.jpg'
file = os.path.join(image_dir, image_file)
image = io.imread(file)
plt.imshow(image)
plt.show()

# draw all files by pattern
files = io.ImageCollection(image_dir + '*.jpg')
for image in files:
    plt.imshow(image)
    plt.show()
