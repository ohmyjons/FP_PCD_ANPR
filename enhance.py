import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import imutils

# load citra
img = cv.imread(r"test images\1.jpg") # plat nomer not detect
# img = cv.imread(r"test images\AB2638XU.jpg")

# img = cv.imread(r"test images\plat2.jpeg") # plat nomer not detect
img = imutils.resize(img,width=1280)
img = cv.cvtColor(img,cv.COLOR_BGR2GRAY)

# thresh
blur = cv.GaussianBlur(img,(5,5),0)
ret,th1 = cv.threshold(blur,0,255, cv.THRESH_BINARY + cv.THRESH_OTSU)

th2 = cv.adaptiveThreshold(img,255,cv.ADAPTIVE_THRESH_MEAN_C ,cv.THRESH_BINARY,11,2)

th3 = cv.adaptiveThreshold(img,127,cv.ADAPTIVE_THRESH_GAUSSIAN_C ,cv.THRESH_BINARY,11,2)


# show image
fig = plt.figure(figsize=(10, 7))
row_fig = 2
column_fig = 2

fig.add_subplot(row_fig, column_fig, 1)
plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
plt.axis('on')
plt.title("RGB")

fig.add_subplot(row_fig, column_fig, 2)
plt.imshow(th1, cmap='gray')
plt.axis('on')
plt.title("biner")

fig.add_subplot(row_fig, column_fig, 3)
plt.imshow(th2, cmap='gray')
plt.axis('off')
plt.title(" mean")

fig.add_subplot(row_fig, column_fig, 4)
plt.imshow(th3, cmap='gray')
plt.axis('off')
plt.title("Gasussian")

# plt.show()


# tophat

kernel = np.ones((20,20), np.uint8)
tophat = cv.morphologyEx(img,cv.MORPH_BLACKHAT, kernel)

# otsu
ret,th1 = cv.threshold(tophat,0,255, cv.THRESH_BINARY + cv.THRESH_OTSU)


fig = plt.figure(figsize=(10, 7))
row_fig = 2
column_fig = 2

fig.add_subplot(row_fig, column_fig, 1)
plt.imshow(cv.cvtColor(tophat, cv.COLOR_BGR2RGB))
plt.axis('on')
plt.title("tophat")

fig.add_subplot(row_fig, column_fig, 2)
plt.imshow(th1, cmap='gray')
plt.axis('on')
plt.title("otsu")

fig.add_subplot(row_fig, column_fig, 3)
plt.imshow(th2, cmap='gray')
plt.axis('off')
plt.title(" mean")

fig.add_subplot(row_fig, column_fig, 4)
plt.imshow(th3, cmap='gray')
plt.axis('off')
plt.title("Gasussian")

plt.show()