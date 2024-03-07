import cv2
import numpy as np

I_color = cv2.imread('damavand.jpg')
I_gray = I_color.mean(axis=2).astype(np.uint8)
I_gray = np.stack((I_gray, I_gray, I_gray), axis=2)

for i in np.linspace(0, 1, 100):
    K = cv2.addWeighted(I_color, i, I_gray, 1-i, 0)
    cv2.imshow("Blending", K)
    cv2.waitKey(15)

cv2.destroyAllWindows()