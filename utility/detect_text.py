import cv2
import numpy as np
import pytesseract
from fuzzywuzzy import fuzz
import re
from template_maker import Selector

AADHAR_TYPE = 0
SALARY_SLIP = 1
MOBILE_BILL = 2
DEPOSIT_FORM = 3

def convertPdfToImage(pdfFile):
    from wand.image import Image
    from wand.color import Color

    with Image(filename=pdfFile, resolution=300) as img:
        with Image(width=img.width, height=img.height, background=Color("white")) as bg:
            bg.composite(img, 0, 0)
            bg.save(filename="voucher_fill_half.jpg")
def junkRemoval(r,w,h,crop_mask,contours,idx,docType):
    if(docType == AADHAR_TYPE):
        if r > 0.45 and w > 25 and h > 5 and h < 45:
            cv2.rectangle(rgb, (x, y), (x+w-1, y+h-1), (0, 255, 0), 2)
            cv2.drawContours(crop_mask,contours,idx,(255,255,255),-1)
    elif(docType == SALARY_SLIP):
        cv2.rectangle(rgb, (x, y), (x + w - 1, y + h - 1), (0, 255, 0), 2)
        cv2.drawContours(crop_mask, contours, idx, (255, 255, 255), -1)
    elif(docType == MOBILE_BILL):
        if r > 0.45 and w > 25 and h > 5 and h < 45:
            cv2.rectangle(rgb, (x, y), (x+w-1, y+h-1), (0, 255, 0), 2)
            cv2.drawContours(crop_mask,contours,idx,(255,255,255),-1)
    elif(docType == DEPOSIT_FORM):
        if r > 0.25 and w > 10 and h > 10 and h < 45:
            cv2.rectangle(rgb, (x, y), (x+w-1, y+h-1), (0, 255, 0), 2)
            cv2.drawContours(crop_mask,contours,idx,(255,255,255),-1)
        # cv2.rectangle(rgb, (x, y), (x + w - 1, y + h - 1), (0, 255, 0), 2)
        # cv2.drawContours(crop_mask, contours, idx, (255, 255, 255), -1)

def binaryTunning(image_mask,docType):
    if(docType == AADHAR_TYPE):
        # pay_slip 125,125,125
        mser_extracted_text[np.where((mser_extracted_text <= [125, 125, 125]).all(axis=2))] = [255, 255, 255]
        mser_extracted_text[np.where((mser_extracted_text <= [254, 254, 254]).all(axis=2))] = [0, 0, 0]
    elif(docType == SALARY_SLIP):
        # pay_slip 70,70,70
        mser_extracted_text[np.where((mser_extracted_text <= [70, 70, 70]).all(axis=2))] = [255, 255, 255]
        mser_extracted_text[np.where((mser_extracted_text <= [254, 254, 254]).all(axis=2))] = [0, 0, 0]
    elif(docType == MOBILE_BILL):
        # pay_slip 90,90,90
        mser_extracted_text[np.where((mser_extracted_text <= [90, 90, 90]).all(axis=2))] = [255, 255, 255]
        mser_extracted_text[np.where((mser_extracted_text <= [254, 254, 254]).all(axis=2))] = [0, 0, 0]
    elif(docType == DEPOSIT_FORM):
        # pay_slip 90,90,90
        mser_extracted_text[np.where((mser_extracted_text <= [90, 90, 90]).all(axis=2))] = [255, 255, 255]
        mser_extracted_text[np.where((mser_extracted_text <= [254, 254, 254]).all(axis=2))] = [0, 0, 0]


def performMSER():
    mser = cv2.MSER_create()
    regions, _ = mser.detectRegions(text_only_adaptive)
    hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in regions]
    # hulls = regions
    # vis = small.copy()
    # cv2.polylines(vis, hulls, False, (0, 255, 0))
    # cv2.imshow("vis",vis)
    mser_mask = np.zeros(text_only_adaptive.shape, dtype=np.uint8)
    mser_mask_temp = np.zeros(text_only_adaptive.shape, dtype=np.uint8)

    for idx in range(len(hulls)):
        x, y, w, h = cv2.boundingRect(hulls[idx])
        mser_mask_temp[y:y + h, x:x + w] = 0
        cv2.drawContours(mser_mask_temp, hulls, idx, (255, 255, 255), -1)
        r = float(cv2.countNonZero(mser_mask_temp[y:y + h, x:x + w])) / (w * h)

        # Tune Below code based on document font and layout
        junkRemoval(r, w, h, mser_mask, hulls, idx, document_type)
        # if r > 0.45 and w > 25 and h > 5 and h < 45:
        #     cv2.drawContours(mser_mask,hulls,idx,(255,255,255),-1)
    return mser_mask


def getRegionOfintrest(path):
    selected_template_regions =[[(168, 92), (191, 114)], [(198, 91), (219, 112)], [(229, 94), (249, 115)], [(258, 94), (282, 115)], [(290, 94), (310, 112)], [(318, 94), (339, 115)], [(348, 94), (370, 114)], [(370, 114), (370, 114)], [(379, 94), (397, 115)], [(407, 94), (429, 115)], [(439, 95), (462, 115)], [(469, 95), (490, 114)], [(497, 94), (519, 114)], [(527, 94), (546, 115)], [(556, 95), (577, 114)], [(590, 97), (611, 115)]]

    # selected_template_regions = Selector.select(path)
    # print(selected_template_regions)
    temp = cv2.imread('aligned.jpg')
    i=0
    for r in selected_template_regions:
        roi = temp[r[0][1]:r[1][1], r[0][0]:r[1][0]]
        try:
            cv2.imwrite("chardata/temp"+str(i)+".jpg",roi)
        except:
            print("wrong region")
        i = i+1
        print(pytesseract.image_to_string(roi,config='--oem 1'))

# convertPdfToImage("voucher_fill_half.pdf")
# getRegionOfintrest("aligned.jpg")
document_type = DEPOSIT_FORM
word_to_search = 'pan'
large = cv2.imread('aligned.jpg')
# large = cv2.resize(large,None,fx=0.5, fy=0.5, interpolation = cv2.INTER_LINEAR)
rgb = cv2.pyrDown(large)
img = rgb.copy()
small = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
grad = cv2.morphologyEx(small, cv2.MORPH_GRADIENT, kernel)

_, bw = cv2.threshold(grad, 0.0, 255.0, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 1))
connected = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel)
# using RETR_EXTERNAL instead of RETR_CCOMP
(_,contours, hierarchy) = cv2.findContours(connected.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
# (_,cnts, _)
mask = np.zeros(bw.shape, dtype=np.uint8)
crop_mask = np.zeros(bw.shape,dtype=np.uint8)
for idx in range(len(contours)):
    x, y, w, h = cv2.boundingRect(contours[idx])
    mask[y:y+h,x:x+w] = 0
    cv2.drawContours(mask, contours, idx, (255, 255, 255), -1)
    r = float(cv2.countNonZero(mask[y:y+h, x:x+w])) / (w * h)

    # Tune Below code based on document font and layout
    junkRemoval(r,w,h,crop_mask,contours,idx,document_type)
    # if r > 0.45 and w > 25 and h > 5 and h < 45:
    #     cv2.rectangle(rgb, (x, y), (x+w-1, y+h-1), (0, 255, 0), 2)
    #     cv2.drawContours(crop_mask,contours,idx,(255,255,255),-1)

extract_text = cv2.bitwise_and(small,small,mask=crop_mask)
text_only_adaptive = cv2.adaptiveThreshold(extract_text,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,17,2)


# cv2.imshow("mser_mask",performMSER())
if(document_type != SALARY_SLIP):
    mser_extracted_text = cv2.bitwise_and(img, img, mask=performMSER())
    mser_extracted_text = cv2.bitwise_not(mser_extracted_text)
else:
    mser_extracted_text = cv2.bitwise_and(img, img, mask=crop_mask)
    mser_extracted_text = cv2.bitwise_not(mser_extracted_text)
# mser_text_only_adaptive = cv2.adaptiveThreshold(mser_extracted_text,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,17,2)

#pay_slip 90,90,90
binaryTunning(mser_extracted_text,document_type)
# mser_extracted_text[np.where((mser_extracted_text <= [125,125,125]).all(axis = 2))] = [255,255,255]
# mser_extracted_text[np.where((mser_extracted_text <= [254,254,254]).all(axis = 2))] = [0,0,0]

# kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
# mser_extracted_text = cv2.morphologyEx(mser_extracted_text,cv2.MORPH_OPEN,kernel)

# text = pytesseract.image_to_data(mser_extracted_text,config='--oem 1 --psm 11',lang='eng+hin')
# print(text)

dict = pytesseract.image_to_data(mser_extracted_text,config='--oem 1 --psm 11', output_type = pytesseract.Output.DICT)
# print(dict)
result = ""
matching_ratio = 0
searched_word_flag = False
previous_words = {}
number_flag = False
# previous_words[0] = ''
searched_word_list = word_to_search.split(' ')
for word,word_num,conf,left,top,width,height in zip(dict['text'],dict['word_num'],dict['conf'],dict['left'],dict['top'],dict['width'],dict['height']):
    if(searched_word_flag == True):
        if(word_num != 0):
            # print("word_num",word_num,word)
            numberList = re.findall("\d+\.?\d+",word)
            if(len(numberList)>0 and numberList[0].replace('.','',1).isdigit()):
                # searched_word_flag = False
                number_flag = True
            if(not(len(numberList)>0 and numberList[0].replace('.','',1).isdigit()) and number_flag):
                searched_word_flag = False
                continue
            cv2.rectangle(rgb, (left, top), (left + width, top + height), (255, 0, 0), 2)
            continue
        else:
            # print("word_num",word_num,word)
            searched_word_flag = False
            # matching_ratio = 0
    if(conf > 60 and word != ' '):
        if(len(searched_word_list) > 0):
            # print(len(searched_word_list))
            # print(searched_word_list)
            if(fuzz.ratio(searched_word_list[0],word.lower()) > 80):
                # previous_words[1] = previous_words[0]
                # print(word)
                previous_words[len(previous_words)] = (word,left,top,width,height)
                # print(previous_words,len(previous_words))
                searched_word_list.pop(0)
                # print(len(searched_word_list))

        if(len(previous_words) > 0 and len(searched_word_list)==0):
            print(len(previous_words))
            for word_tuple in range(0,len(previous_words)):
                print(previous_words[word_tuple])
                word,left,top,width,height = previous_words[word_tuple]
                cv2.rectangle(rgb, (left, top), (left + width, top + height), (255, 0, 0), 2)
            searched_word_flag = True
            previous_words.clear()
            continue
        # if(len(word) >= len(word_to_search)):
        #     if(word_to_search in word.lower()):
        #         matching_ratio = matching_ratio+fuzz.ratio(word_to_search, word.lower())
        # else:
        #     if(word.lower() in word_to_search.lower()):
        #         matching_ratio = matching_ratio + fuzz.ratio(word.lower(),word_to_search.lower())
        # if(matching_ratio >= 90):
        #     print(word,word_num,previous_words,matching_ratio)
        #     cv2.rectangle(rgb, (left, top), (left + width, top + height), (255, 0, 0), 2)
        #     searched_word_flag = True
        #     matching_ratio = 0
        # else:
        cv2.rectangle(rgb, (left, top), (left+width, top+height), (0, 0, 255), 2)
        # print(word,conf,left,top,width,height)


cv2.imshow("text_only_adaptive",text_only_adaptive)
cv2.imshow("mser_extracted_text",mser_extracted_text)
# cv2.imshow("mser_text_only_adaptive",mser_text_only_adaptive)
img_resize = cv2.resize(rgb,None,fx=0.7, fy=0.7, interpolation = cv2.INTER_LINEAR)
cv2.imshow('rects', img_resize)
# cv2.imshow('canny', canny)

# cv2.imwrite("mser_text_only_adaptive.jpg",mser_text_only_adaptive)
cv2.imwrite('extract_text.jpg',extract_text)
cv2.imwrite('mser_extracted_text.jpg',mser_extracted_text)

cv2.imwrite('temp.jpg',rgb)
cv2.waitKey(0)
