import numpy as np
import cv2

#img = cv2.imread('rtg_examples/pics/healthy/healthy_32yo_male_RTG.jpg',0)
#img = cv2.imread('rtg_examples/pics/healthy/healthy_75yo_female_RTG.jpg',0)
img = cv2.imread('rtg_examples/pics/COVID-19/covid19_30yo_female_RTG.jpeg', 0)

cv2.imshow('aa',img)

equ = cv2.equalizeHist(img)
res = np.hstack((img,equ)) #stacking images side-by-side
cv2.imshow('compare',res)

blur = cv2.blur(equ,(5,5))
cv2.imshow('blur',blur)

cany = cv2.Canny(blur, 20, 50, None,3)
cv2.imshow('Canny',cany)

kernel = np.ones((5,5),np.uint8)
erosion = cv2.erode(cany,kernel,iterations = 1)
cv2.imshow('ero', erosion)

opening = cv2.morphologyEx(cany, cv2.MORPH_OPEN, kernel)
cv2.imshow('ero', opening)

dilation = cv2.dilate(cany,kernel,iterations = 4)
cv2.imshow('dilo', dilation)

erosion2 = cv2.erode(dilation,kernel,iterations = 6)
cv2.imshow('eroafter opening', erosion2)

closing2 = cv2.morphologyEx(erosion2, cv2.MORPH_CLOSE, kernel)
cv2.imshow('eroafter opening closing2', closing2)

closing3 = cv2.morphologyEx(closing2, cv2.MORPH_CLOSE, kernel)
cv2.imshow('eroafter opening closing3', closing3)

erosion3 = cv2.erode(closing3,kernel,iterations =3)
#cv2.imshow('eroafter opening3', erosion3)

closing4 = cv2.morphologyEx(erosion3, cv2.MORPH_CLOSE, kernel)
#cv2.imshow('eroafter opening4', closing4)

cany2b = cv2.Canny(closing3, 40, 100, None,5)
# cv2.imshow('Cannyclosing3b',cany2b)

sample = cany2b.copy()

contours, hierarchy = cv2.findContours(cany2b, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# cv2.drawContours(sample, contours, -1, (0,255,0), 3)

cnt = contours[4]
no_of_contours = len(contours)
cv2.drawContours(sample, contours, no_of_contours-1, (0,255,0), 3)
cv2.imshow('with contours',sample)

M = cv2.moments(sample)

# calculate x,y coordinate of center
cX = int(M["m10"] / M["m00"])
cY = int(M["m01"] / M["m00"])
# put text and highlight the center
cv2.circle(sample, (cX, cY), 5, (255, 255, 255), -1)
cv2.putText(sample, "centroid", (cX - 25, cY - 25),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)


# cv2.circle(sample, (50, 100), 5, (255, 255, 255), -1)
# cany3b = cv2.Canny(sample, 40, 100, None,5)
cv2.imshow('Cannyclosing3bb',sample)

m,n = sample.shape[:2]

distance_x=[]
distance_y=[]

for i in range(0,m):
    for j in range(0,n):
        if sample[i,j] > 128:
            distance_x.append(abs(cX-i))
            distance_y.append(abs(cY-j))

distance_x.sort()
distance_y.sort()

maxX = distance_x[-1]
maxY = distance_y[-1]

print (maxX)
deltaX = int(0.7*maxX)
deltaY = int(0.7*maxY)

image = cv2.rectangle(sample, (cX-deltaX,cY-deltaY), (cX+deltaX,cY+deltaY), (255,0,0), thickness=2)
# cv2.imshow('lung',image)
# print(distance_x)
# 
image_with_rect = cv2.rectangle(img, (cX-deltaX,cY-deltaY), (cX+deltaX,cY+deltaY), (255,0,0), thickness=2)
cv2.imshow('lung',image_with_rect)
# erosion3 = cv2.erode(cany2,kernel,iterations = 6)
# cv2.imshow('eroafter canny', erosion3)

cv2.waitKey(0)
cv2.destroyAllWindows()

# images = glob.glob('calib_example/*.tif')
# for name in images: