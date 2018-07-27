import cv2
import numpy as np
from PIL import Image
from PIL import ImageChops

refPt = []
selectedReagon = []
cropping = False


class Selector():
    def __init__(self):
        print("constructor")

    def click_and_crop(event, x, y, flags, param):
        # grab references to the global variables
        global selectedReagon, refPt, cropping

        # if the left mouse button was clicked, record the starting
        # (x, y) coordinates and indicate that cropping is being
        # performed

        if event == cv2.EVENT_LBUTTONDOWN and cropping == False:
            refPt = [(x, y)]
            cropping = True

        # check to see if the left mouse button was released
        elif event == cv2.EVENT_LBUTTONUP and cropping == True:
            # record the ending (x, y) coordinates and indicate that
            # the cropping operation is finished
            refPt.append((x, y))
            cropping = False
            # draw a rectangle around the region of interest
            cv2.rectangle(img, refPt[0], refPt[1], (0, 0, 255), 1)
            cv2.imshow("image", img)
            selectedReagon.append(refPt)
            refPt = []
            cropping = False

        elif event == cv2.EVENT_LBUTTONUP:
            cropping = False

        elif event == cv2.EVENT_MOUSEMOVE and cropping == True:
            cv2.rectangle(img, refPt[0], (x, y), (0, 255, 0), 1)

    def process_image(img):
        cv2.namedWindow("image", cv2.WINDOW_FULLSCREEN)
        # cv2.resizeWindow("image", 650, 750)
        cv2.setMouseCallback("image", Selector.click_and_crop)
        # image = image.copy()
        while True:
            cv2.imshow("image", img)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("r"):
                img = clone.copy()
                # if the 'c' key is pressed, break from the loop
            if (key == ord("c")):
                # selectedReagon = click_and_crop()
                # save_template_crop(selectedReagon)
                # print(selectedReagon)
                break
            if (key == ord("d")):
                break

    def select(image_path):

        # Read image
        global img, clone
        img1 = cv2.imread(image_path)
        img = img1.copy()

        # img = cv2.resize(img,None,fx=0.2, fy=0.2, interpolation = cv2.INTER_LINEAR)

        clone = img.copy()
        # Select ROI
        # r = cv2.selectROI(img)
        Selector.process_image(img)
        # return selectedReagon;
        # cv2.setMouseCallback("image", click_and_crop)

        # print(r)

        # Crop image
        # imCrop = img[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]

        # Display cropped image
        # cv2.imshow("Image", imCrop)
        # cv2.waitKey(0)
        return selectedReagon

    def crop_pil(gray_image_path, origional_image_path):
        im = Image.open(gray_image_path)
        im_origional = Image.open(origional_image_path)
        # size = 200,300
        # im.thumbnail(size, Image.ANTIALIAS)
        # im = cv2.imread("panbothside_test1.jpg")
        bg = Image.new(im.mode, im.size, im.getpixel((0, 0)))

        # bg.show()

        # box = (0, 0, im.size[0], im.size[1]) # The box containing the original image
        diff = ImageChops.difference(im, bg)
        # diff = ImageChops.add(diff, diff, 2.0, -100)

        diff = ImageChops.add(diff, diff)

        bbox = diff.getbbox()
        # bbox= (bbox[0] - margin,) + bbox[1:2] + (bbox[2]+margin,)+ bbox[3:]
        # print(bbox)
        if bbox:
            crop_img = im_origional.crop(bbox)
            # size = 1200,890
            # crop_img.thumbnail(size, Image.ANTIALIAS)
            # crop_img = crop_img.resize(size, Image.ANTIALIAS)
            crop_img.save("cropped.jpg")
            # crop_img.show()
if __name__=='__main__':
    regions = Selector.select("template_half.jpg")
    print(regions)