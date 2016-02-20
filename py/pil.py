from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

img = Image.open("../images/wheat.jpg")
gray = img.convert('L')   # 'L' stands for 'luminosity'
gray = np.asarray(gray)

plt.imshow(img)
plt.show()

plt.imshow(gray)
plt.show()

# Plot each color separately:

r, g, b = np.rollaxis(img, -1)
plt.contour(r, cmap=plt.cm.Reds)
plt.contour(g, cmap=plt.cm.Greens)
plt.contour(b, cmap=plt.cm.Blues)
plt.show()