import numpy as np
import cv2

img = cv2.imread('rtg_examples/pics/healthy/healthy_32yo_male_RTG.jpg',0)
#img = cv2.imread('rtg_examples/pics/healthy/healthy_75yo_female_RTG.jpg',0)
#img = cv2.imread('rtg_examples/pics/COVID-19/covid19_30yo_female_RTG.jpeg', 0)

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
#cv2.imshow('eroafter opening', erosion2)

closing2 = cv2.morphologyEx(erosion2, cv2.MORPH_CLOSE, kernel)
# cv2.imshow('eroafter opening closing2', closing2)

closing3 = cv2.morphologyEx(closing2, cv2.MORPH_CLOSE, kernel)
#cv2.imshow('eroafter opening closing3', closing3)

erosion3 = cv2.erode(closing3,kernel,iterations =3)
# cv2.imshow('eroafter opening3', erosion3)

closing4 = cv2.morphologyEx(erosion3, cv2.MORPH_CLOSE, kernel)
# cv2.imshow('eroafter opening4', closing4)

cv2.waitKey(0)
cv2.destroyAllWindows()

# images = glob.glob('calib_example/*.tif')
# for name in images: