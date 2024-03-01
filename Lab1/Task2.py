from matplotlib import pyplot as plt
import numpy as np

I = plt.imread('masoleh_gray.jpg')

I_v_inv = I[::-1, :]
new_image = np.vstack((I, I_v_inv))
plt.imshow(new_image, cmap='gray')
plt.show()