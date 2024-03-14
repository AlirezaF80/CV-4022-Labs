import cv2
import numpy as np
from matplotlib import pyplot as plt

def apply_intensity_windowing(I, a, b):
    f, axes = plt.subplots(2, 3)

    axes[0,0].imshow(I, 'gray', vmin=0, vmax=255)
    axes[0,0].axis('off')

    axes[1,0].hist(I.ravel(),256,[0,256])

    J = (I-a) * 255.0 / (b-a)
    J[J < 0] = 0
    J[J > 255] = 255
    J = J.astype(np.uint8)

    axes[0,1].imshow(J, 'gray', vmin=0, vmax=255)
    axes[0,1].axis('off')

    axes[1,1].hist(J.ravel(),256,[0,256])

    K = cv2.equalizeHist(I)

    axes[0,2].imshow(K, 'gray', vmin=0, vmax=255)
    axes[0,2].axis('off')

    axes[1,2].hist(K.ravel(),256,[0,256])

    plt.show()

def get_contrast_stretching_params(I):
    I_ravel = I.ravel()
    return np.percentile(I_ravel, 5), np.percentile(I_ravel, 95)

crayfish_I = cv2.imread('crayfish.jpg', cv2.IMREAD_GRAYSCALE)
a, b = get_contrast_stretching_params(crayfish_I)
print(f'crayfish: a={a}, b={b}')
# apply_intensity_windowing(crayfish_I, 100, 200)
apply_intensity_windowing(crayfish_I, a, b)
map_I = cv2.imread('map.jpg', cv2.IMREAD_GRAYSCALE)
a, b = get_contrast_stretching_params(map_I)
print(f'map: a={a}, b={b}')
# apply_intensity_windowing(map_I, 150, 215)
apply_intensity_windowing(map_I, a, b)
train_I = cv2.imread('train.jpg', cv2.IMREAD_GRAYSCALE)
a, b = get_contrast_stretching_params(train_I)
print(f'train: a={a}, b={b}')
# apply_intensity_windowing(train_I, 70, 235)
apply_intensity_windowing(train_I, a, b)