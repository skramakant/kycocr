import cv2
import numpy as np

#Use Image Magic Command below
# convert crop.jpg -type Grayscale -negate -define morphology:compose=darken -morphology Thinning 'Rectangle:1x60+0+0<' -negate mgiccrop.jpg

#or
#convert crop.jpg
# \( -clone 0 -threshold 50% -negate -statistic median 150x1
# \) -compose lighten -composite
# \( -clone 0 -threshold 50% -negate -statistic median 1x150
# \) -composite result.jpg\

#links
#https://stackoverflow.com/questions/33949831/whats-the-way-to-remove-all-lines-and-borders-in-imagekeep-texts-programmatic


def performMSER(bw):

    mser = cv2.MSER_create()
    regions, _ = mser.detectRegions(bw)
    hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in regions]
    # hulls = regions
    vis = bw.copy()
    # cv2.polylines(vis, hulls, False, (0, 0, 255))

    mser_mask = np.full(bw.shape, 255,dtype=np.uint8)
    mser_mask_temp = np.full(bw.shape,255, dtype=np.uint8)
    cv2.drawContours(mser_mask, hulls, -1, (0, 0, 255), 2)
    cv2.imshow("vis", mser_mask)
    for idx in range(len(hulls)):
        x, y, w, h = cv2.boundingRect(hulls[idx])
        # mser_mask_temp[y:y + h, x:x + w] = 0
        # cv2.drawContours(mser_mask_temp, hulls, idx, (255,255, 255), 2)
        # r = float(cv2.countNonZero(mser_mask_temp[y:y + h, x:x + w])) / (w * h)

        # Tune Below code based on document font and layout
        # junkRemoval(r, w, h, mser_mask, hulls, idx, document_type)
        # if w > 10 and w < 20 and h > 10 and h < 25:
        #     cv2.drawContours(mser_mask_temp,hulls,idx,(0,0,0),1)
        #     mser_mask_temp[y:y + h, x:x + w] = 0

        if w > 20 and h > 20:
            cv2.drawContours(mser_mask_temp,hulls,idx,(0,0,0),6)
            # mser_mask_temp[y:y + h, x:x + w] = 0
    return mser_mask_temp

img = cv2.imread('bc.jpg')
cv2.imshow("bc",img)
# edges = cv2.Canny(img,80,10,apertureSize = 3) # canny Edge OR
#
# # Hough's Probabilistic Line Transform
# minLineLength = 900
# maxLineGap = 100
# lines = cv2.HoughLinesP(edges,1,np.pi/180,100,minLineLength,maxLineGap)
# for line in lines:
#     for x1,y1,x2,y2 in line:
#         cv2.line(img,(x1,y1),(x2,y2),(255,255,255),2)
#
# cv2.imwrite('houghlines.jpg',img)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, bw = cv2.threshold(gray, 0.0, 255.0, cv2.THRESH_BINARY | cv2.THRESH_OTSU)


rows,columns,channel = img.shape
print(img.shape,img[0,0])



return_mask = performMSER(bw)
print(return_mask.shape,return_mask[0,0])
row = 0
column = 0
abc = img.copy()
for row in range(rows):
    for column in range(columns):
        if np.where((img[row, column] < [150,150,150])) and (return_mask[row, column] < 50):
            # return_mask[row][column] = 205
            abc[row][column] = [205,205,205]

# img =  cv2.bitwise_not(img)
# extract_text = cv2.bitwise_and(img,img,mask=return_mask)
#
abc[np.where((abc <= [205, 205, 205]).all(axis=2)) and np.where((abc >= [170, 170, 170]).all(axis=2))] = [255, 255, 255]

cv2.imshow('img',abc)
cv2.imwrite('temp.jpg',abc)
# cv2.imshow('text',extract_text)
cv2.imshow('mask',return_mask)



horizontal = bw.copy()
vertical = bw.copy()
print(horizontal.shape)
rows,columns = horizontal.shape
hscale = 15
horizontalSize = (columns/hscale)

horizontalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (int(horizontalSize), 1))
horizontal = cv2.erode(horizontal,horizontalStructure,(-1,-1))
horizontal = cv2.dilate(horizontal,horizontalStructure,(-1,-1))

# cv2.imshow("horizontal",horizontal)

vscale = 45
verticalSize = rows/vscale
verticalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (1,int(verticalSize)))
vertical = cv2.erode(vertical,verticalStructure,(-1,-1))
vertical = cv2.dilate(vertical,verticalStructure,(-1,-1))

# cv2.imshow("vertical",vertical)

result = horizontal + vertical

# cv2.imshow("result",result)

cv2.waitKey(0)