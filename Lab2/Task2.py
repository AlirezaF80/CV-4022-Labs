import cv2
import numpy as np

I = cv2.imread('damavand.jpg')
J = cv2.imread('eram.jpg')
print(I.shape)
print(J.shape)

for i in np.linspace(0, 1, 100):
    K = cv2.addWeighted(I, i, J, 1-i, 0)
    cv2.imshow("Blending", K)
    cv2.waitKey(15)

cv2.destroyAllWindows()