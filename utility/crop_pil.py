from PIL import Image
from PIL import ImageChops
import numpy as np
import cv2

im1 = Image.open("voucher_fill_half.jpg")
im = im1.copy()
# im=im.convert("L")
threshold = 20
im = im.point(lambda p: p > threshold and 255)
im.save("gray.jpg")
# size = 200,300
# im.thumbnail(size, Image.ANTIALIAS)
# im = cv2.imread("panbothside_test1.jpg")
bg = Image.new(im.mode, im.size, im.getpixel((0, 0)))

# bg.show()

# box = (0, 0, im.size[0], im.size[1]) # The box containing the original image
diff = ImageChops.difference(im, bg)
# diff = ImageChops.add(diff, diff, 2.0, -100)

# diff = ImageChops.add(diff, diff)

bbox = diff.getbbox()
#bbox= (bbox[0] - margin,) + bbox[1:2] + (bbox[2]+margin,)+ bbox[3:]
print(bbox)
if bbox:
	crop_img = im1.crop(bbox)
	crop_img.save("crop.jpg")
	# crop_img.show()
	# cv2.imshow("crop_imag",crop_img)

# cv2.waitKey(0)