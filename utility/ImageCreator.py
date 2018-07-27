import cv2
import numpy as np

def creator():
    origional_image = cv2.imread('date.jpg')
    print(origional_image.shape)
    rows, cols, channel = origional_image.shape
    px = origional_image[1,1]
    print(px)
    background = cv2.imread('background.jpg')
    print(background.shape)
    # background[np.where((background >= [230, 230, 230]).all(axis=2))] = px
    cv2.imshow("back",background)
    # dst = background + origional_image
    background[150:rows+150, 160:cols+160] = origional_image
    # background = cv2.bitwise_not(background)
    # background[np.where((background <= [100, 100, 100]).all(axis=2))] = [255,255,255]
    # background[np.where((background <= [150, 150, 150]).all(axis=2))] = [0,0,0]


    cv2.imwrite('bc.jpg',background)
    # dst = cv2.addWeighted(background, 0.7, origional_image, 0.3, 0)
    # cv.imshow('dst', dst)

    # origional_image = cv2.cvtColor(origional_image,cv2.COLOR_BGR2GRAY)
    # cv2.imshow("oregional",origional_image)
    # height,width = origional_image.shape
    # background = np.zeros((height*4,width*2),np.uint8)
    # background = cv2.bitwise_not(background)
    # h,w = background.shape
    # cv2.imshow("background",background)
    #
    # dst = cv2.addWeighted(background, origional_image)
    cv2.imshow("final",background)
    cv2.waitKey(0)

if __name__== '__main__':
    creator()
    # cv2.waitKey(0)
