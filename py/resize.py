from skimage import transform
from skimage import io
import matplotlib.pyplot as plt
from scipy import misc

image_dir = "../images/examples/15p/"
files = io.ImageCollection(image_dir + '*.JPG')

for f in files.files:
    image = misc.imread(f)
    print(image.shape)
    #plt.imshow(image)
    #plt.show()
    rescaled = transform.rescale(image, 0.25)
    io.imsave(f, rescaled)
    print("Image rescaled")
