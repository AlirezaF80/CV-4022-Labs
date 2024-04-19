import numpy as np
import cv2

def std_filter(I, ksize):
    F = np.ones((ksize,ksize), dtype=np.float) / (ksize*ksize) # a box filter
       
    MI = cv2.filter2D(I,-1,F) # apply mean filter on I

    I2 = I * I # I squared
    MI2 = cv2.filter2D(I2,-1,F) # apply mean filter on I2

    return np.sqrt(MI2 - MI * MI)

def zero_crossing(I):
    """Finds locations at which zero-crossing occurs, used for
    Laplacian edge detector"""
    
    Ishrx = I.copy() # shift right
    Ishrx[:,1:] = Ishrx[:,:-1]
        
    Ishdy = I.copy() # shift down
    Ishdy[1:,:] = Ishdy[:-1,:]
        
    ZC = (I==0) | (I * Ishrx < 0) | (I * Ishdy < 0) # zero crossing locations
    # I * Ishrx < 0 is true when the laplacian changes sign in the x direction
    # I * Ishdy < 0 is true when the laplacian changes sign in the y direction

    SI = std_filter(I, 3) / I.max()

    Mask =  ZC & (SI > .1)

    E = Mask.astype(np.uint8) * 255 # the edges

    return E

text = ""
position = (10, 30)
font = cv2.FONT_HERSHEY_SIMPLEX
fontScale = 1
fontColor = (255, 255, 255)
lineType = 2

cam_id = 1  # camera id
# for default webcam, cam_id is usually 0
# try out other numbers (1,2,..) if this does not work

cap = cv2.VideoCapture(cam_id)

mode = 'o' # show the original image at the beginning

sigma = 21
sobel_thresh = 60
canny_low = 20
canny_high = 100

while True:
    ret, I = cap.read()
    #I = cv2.imread("agha-bozorg.jpg") # can use this for testing 
    I = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY) # convert to grayscale
    Ib = cv2.GaussianBlur(I, (sigma,sigma), 0) # blur the image
    
    if mode == 'o':
        text = "Original"
        # J = the original image
        J = Ib
    elif mode == 'x':
        text = "Sobel X"
        # J = Sobel gradient in x direction
        J = np.abs(cv2.Sobel(Ib,cv2.CV_64F,1,0))
        
    elif mode == 'y':
        text = "Sobel Y"
        # J = Sobel gradient in y direction
        J = np.abs(cv2.Sobel(Ib,cv2.CV_64F,0,1))
    
    elif mode == 'm':
        text = "Sobel Magnitude"
        # J = magnitude of Sobel gradient
        S_x = cv2.Sobel(Ib,cv2.CV_64F,1,0)
        S_y = cv2.Sobel(Ib,cv2.CV_64F,0,1)
        J = np.sqrt(S_x**2 + S_y**2)
    
    elif mode == 's':
        text = "Sobel + Thresholding"
        # J = Sobel + thresholding edge detection
        S_x = cv2.Sobel(Ib,cv2.CV_64F,1,0)
        S_y = cv2.Sobel(Ib,cv2.CV_64F,0,1)
        m = np.sqrt(S_x**2 + S_y**2)
        J = np.uint8(m > sobel_thresh) * 255 # thresholding the gradient magnitude
    
    elif mode == 'l':
        text = "Laplacian"
        # J = Laplacian edges
        J = cv2.Laplacian(Ib,cv2.CV_64F,ksize=5)
        J = zero_crossing(J)

    elif mode == 'c':
        text = "Canny"
        # J = Canny edges
        J = cv2.Canny(Ib, canny_low, canny_high)
    
    cv2.putText(J, text, position, font, fontScale, fontColor, lineType)
    
    # we set the image type to float and the
    # maximum value to 1 (for a better illustration)
    # notice that imshow in opencv does not automatically
    # map the min and max values to black and white. 
    J = J.astype(np.float) / J.max()
    cv2.imshow("my stream", J)

    key = chr(cv2.waitKey(1) & 0xFF)

    if key in ['o', 'x', 'y', 'm', 's', 'c', 'l']:
        mode = key
    if key == '-' and sigma > 1:
        sigma -= 2
        print("sigma = %d"%sigma)
    if key in ['+','=']:
        sigma += 2    
        print("sigma = %d"%sigma)
    elif key == 'q':
        break

cap.release()
cv2.destroyAllWindows()
