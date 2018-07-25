from __future__ import print_function
import cv2
import numpy as np
from scipy.misc import imread
#perform the prediction
from keras.models import load_model
from keras import backend as K
import os

# os.chdir('/home/ramakant/Desktop/Voucher_OCR/source/core/')
path_model = 'backend/'
path_template = 'backend/'
path_chardump = 'backend/chardump/'
path_imagedump = 'backend/imagedump/'
path_url = 'chardump/'

MAX_FEATURES = 500
GOOD_MATCH_PERCENT = 0.15
i = 0

#Account Number
account_number_rect = [[(165, 92), (193, 117)], [(194, 91), (223, 117)], [(225, 91), (253, 118)], [(255, 91), (282, 118)], [(285, 92), (312, 118)], [(316, 92), (343, 117)], [(345, 93), (373, 119)], [(375, 92), (403, 118)], [(406, 94), (431, 119)], [(435, 93), (462, 119)], [(463, 93), (492, 119)], [(494, 93), (523, 119)], [(525, 95), (554, 119)], [(556, 93), (582, 119)], [(585, 93), (612, 122)]]

#PAN NUMBER
pan_number_rect = [[(165, 128), (193, 156)], [(194, 130), (223, 156)], [(225, 129), (253, 156)], [(254, 129), (283, 156)], [(285, 130), (314, 156)], [(315, 128), (341, 157)], [(345, 130), (372, 157)], [(375, 131), (402, 157)], [(405, 131), (432, 156)], [(434, 130), (462, 158)]]

#Date
date_rect = [[(895, 141), (921, 168)], [(925, 142), (951, 168)], [(955, 142), (982, 169)], [(985, 143), (1012, 169)], [(1014, 144), (1042, 169)], [(1045, 142), (1071, 169)], [(1075, 142), (1102, 169)], [(1104, 142), (1130, 169)]]

#Mobile
mobile_rect = [[(834, 180), (861, 207)],
          [(864, 181), (891, 206)],
          [(894, 182), (921, 206)],
          [(924, 181), (950, 206)],
          [(953, 183), (981, 206)],
          [(984, 180), (1012, 207)],
          [(1013, 183), (1037, 207)],
          [(1043, 182), (1070, 207)],
          [(1074, 182), (1101, 208)],
          [(1104, 181), (1131, 208)]]



def alignImages(im1, im2):
    # Convert images to grayscale
    im1Gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    im2Gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

    # Detect ORB features and compute descriptors.
    orb = cv2.ORB_create(MAX_FEATURES)
    keypoints1, descriptors1 = orb.detectAndCompute(im1Gray, None)
    keypoints2, descriptors2 = orb.detectAndCompute(im2Gray, None)

    # Match features.
    matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
    matches = matcher.match(descriptors1, descriptors2, None)

    # Sort matches by score
    matches.sort(key=lambda x: x.distance, reverse=False)

    # Remove not so good matches
    numGoodMatches = int(len(matches) * GOOD_MATCH_PERCENT)
    matches = matches[:numGoodMatches]

    # Draw top matches
    imMatches = cv2.drawMatches(im1, keypoints1, im2, keypoints2, matches, None)
    img_resize = cv2.resize(imMatches, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)
    # cv2.imshow("draw_matches",img_resize)
    cv2.imwrite(path_imagedump+"matches.jpg", imMatches)

    # Extract location of good matches
    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)

    for i, match in enumerate(matches):
        points1[i, :] = keypoints1[match.queryIdx].pt
        points2[i, :] = keypoints2[match.trainIdx].pt

    # Find homography
    h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)

    # Use homography
    height, width, channels = im2.shape
    im1Reg = cv2.warpPerspective(im1, h, (width, height))

    return im1Reg, h


def charRecognizition(roi):
    # pass
    # compute a bit-wise inversion so black becomes white and vice versa
    # x = np.invert(roi)
    # convert to a 4D tensor to feed into our model
    # roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    # roi = np.invert(roi)
    x = cv2.resize(roi, (28, 28))
    # print(x.shape)
    # cv2.imshow("temp",x)
    # cv2.waitKey(0)
    # CNN MODEL
    x = x.reshape(1, 28, 28, 1)
    #MNIST MODEL
    # x = x.reshape(1, 784).astype('float32') / 255

    model = load_model(path_model + 'MNISTcnn.h5') #MNISTcnn
    out = model.predict(x)
    normed = [i / sum(out[0]) for i in out[0]]
    print(normed)
    # print(out)
    # print(np.array_str(np.argmax(out, axis=1))[1])
    return (np.array_str(np.argmax(out, axis=1))[1])


def extractMobileNumber(final_image):
    extracted_data = []
    cropped_file_path = []
    i=0
    for r in mobile_rect:
        roi = final_image[r[0][1]:r[1][1], r[0][0]:r[1][0]]
        # roi[np.where((roi <= [150, 150, 150]).all(axis=2))] = [255, 255, 255]
        # roi[np.where((roi <= [254, 254, 254]).all(axis=2))] = [0, 0, 0]

        roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        _, roi = cv2.threshold(roi, 127, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        roi = np.invert(roi)

        try:
            temp_file = path_chardump+"mobile"+str(i)+".jpg"
            cv2.imwrite(temp_file,roi)
            cropped_file_path.append(temp_file)
            i = i + 1
            number = charRecognizition(roi)
            extracted_data.append(number)
        except:
            print("wrong region for mobile number")
    return (extracted_data,cropped_file_path)


def extractAccountNumber(final_image):
    extracted_data = []
    cropped_file_path = []
    i=0
    for r in account_number_rect:
        roi = final_image[r[0][1]:r[1][1], r[0][0]:r[1][0]]
        # roi[np.where((roi <= [150, 150, 150]).all(axis=2))] = [255, 255, 255]
        # roi[np.where((roi <= [254, 254, 254]).all(axis=2))] = [0, 0, 0]

        roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        _, roi = cv2.threshold(roi, 127, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        roi = np.invert(roi)

        try:
            temp_file = path_chardump+"account"+str(i)+".jpg"
            cv2.imwrite(temp_file,roi)
            cropped_file_path.append(temp_file)
            i = i + 1
            number = charRecognizition(roi)
            extracted_data.append(number)
        except:
            print("wrong region for mobile number")
    return (extracted_data,cropped_file_path)


def extractDate(final_image):
    extracted_data = []
    cropped_file_path = []
    i=0
    for r in date_rect:
        roi = final_image[r[0][1]:r[1][1], r[0][0]:r[1][0]]
        # roi[np.where((roi <= [150, 150, 150]).all(axis=2))] = [255, 255, 255]
        # roi[np.where((roi <= [254, 254, 254]).all(axis=2))] = [0, 0, 0]

        roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        _, roi = cv2.threshold(roi, 127, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        roi = np.invert(roi)

        try:
            temp_file = path_chardump+"date"+str(i)+".jpg"
            cv2.imwrite(temp_file,roi)
            cropped_file_path.append(temp_file)
            i = i + 1
            number = charRecognizition(roi)
            extracted_data.append(number)
        except:
            print("wrong region for mobile number")
    return (extracted_data,cropped_file_path)

def processIncomingFile(filename):
    refFilename = path_template + "template_half.jpg"
    imReference = cv2.imread(refFilename, cv2.IMREAD_COLOR)

    # Read image to be aligned
    imFilename = "webapp/uploads/" +  filename
    print("Reading image to align : ", imFilename);
    im = cv2.imread(imFilename, cv2.IMREAD_COLOR)

    print("Aligning images ...")
    # Registered image will be resotred in imReg.
    # The estimated homography will be stored in h.
    imReg, h = alignImages(im, imReference)

    # Write aligned image to disk.
    outFilename = path_template +"aligned.jpg"
    print("Saving aligned image : ", outFilename);
    cv2.imwrite(outFilename, imReg)
    # cv2.imshow("outFilename",imReg)
    outFilename = path_imagedump + "aligned.jpg"
    cv2.imwrite(outFilename, imReg)


    final_image = cv2.imread(outFilename)
    # final_image = cv2.cvtColor(final_image, cv2.COLOR_BGR2GRAY)
    # _, final_image = cv2.threshold(final_image, 0.0, 255.0, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    extracted_account_number,cropped_account_file_path = extractAccountNumber(final_image)
    extracted_date,cropped_date_file_path = extractDate(final_image)
    extracted_mobile_number,cropped_mobile_file_path = extractMobileNumber(final_image)
    K.clear_session()
    all_cropped_file_path = []
    all_cropped_file_path = cropped_account_file_path + cropped_date_file_path + cropped_mobile_file_path
    return (extracted_account_number,extracted_date,extracted_mobile_number,all_cropped_file_path)


if __name__ == '__main__':
    # Read reference image
    refFilename = path_template +"template_half.jpg"
    print("Reading reference image : ", refFilename)
    imReference = cv2.imread(refFilename, cv2.IMREAD_COLOR)

    # Read image to be aligned
    imFilename = path_template + "voucher1.jpg"
    print("Reading image to align : ", imFilename);
    im = cv2.imread(imFilename, cv2.IMREAD_COLOR)

    print("Aligning images ...")
    # Registered image will be resotred in imReg.
    # The estimated homography will be stored in h.
    imReg, h = alignImages(im, imReference)

    # Write aligned image to disk.
    outFilename = "aligned.jpg"
    print("Saving aligned image : ", outFilename);
    # cv2.imshow("outFilename",imReg)
    cv2.imwrite(path_imagedump+outFilename, imReg)
    cv2.imwrite(outFilename, imReg)

    final_image = cv2.imread(outFilename)
    # final_image = cv2.cvtColor(final_image, cv2.COLOR_BGR2GRAY)
    # _, final_image = cv2.threshold(final_image, 0.0, 255.0, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    extracted_account_number = extractAccountNumber(final_image)
    extracted_date = extractDate(final_image)
    extracted_mobile_number = extractMobileNumber(final_image)

    print("Account Number: ",extracted_account_number)
    print("date: ", extracted_date)
    print("Mobile Number: ", extracted_mobile_number)
    K.clear_session()
    # Print estimated homography
    # print("Estimated homography : \n", h)
    # cv2.waitKey(0)