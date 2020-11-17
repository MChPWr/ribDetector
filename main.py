import numpy as np
import cv2

#img = cv2.imread('rtg_examples/pics/healthy/healthy_32yo_male_RTG.jpg',0)
img = cv2.imread('rtg_examples/pics/healthy/healthy_75yo_female_RTG.jpg',0)
#img = cv2.imread('rtg_examples/pics/COVID-19/covid19_30yo_female_RTG.jpeg', 0)

cv2.imshow('aa',img)

equ = cv2.equalizeHist(img)
res = np.hstack((img,equ)) #stacking images side-by-side

cv2.imshow('compare',res)

blur = cv2.blur(equ,(5,5))

cv2.imshow('blur',blur)

dst = cv2.Canny(blur, 20, 50, None,3)

cv2.imshow('Canny',dst)

cv2.waitKey(0)
cv2.destroyAllWindows()

# images = glob.glob('calib_example/*.tif')
# for name in images: