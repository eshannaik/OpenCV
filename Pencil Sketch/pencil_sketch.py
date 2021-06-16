import cv2
import numpy as np

lower_range = 0.4
upper_range = 0.7
# imgFileList = ('./girl.jpg','./ferb.png')

def sobel(img):

	x = cv2.Sobel(img,cv2.CV_8U,0,1,ksize=3) #detects horizontal edges
	y = cv2.Sobel(img,cv2.CV_8U,1,0,ksize=3) #detects vertical edges

	return cv2.bitwise_or(x,y) # combine the edges

def sketch(frame):

	frame = cv2.GaussianBlur(frame,(3,3),0) #blur to remove noise
	rImg = 255-frame #black and white image

	#detect edges
	edgImg0 = sobel(frame)
	edgImg1 = sobel(rImg)
	edgImg = cv2.addWeighted(edgImg0,2,edgImg1,5,0)

	# invert the image back 
	opImg = 255-edgImg
	return opImg

if __name__ == '__main__':
	print("Write the name as ./____ if the image is in the same folder")
	i = input("Enter File Location :")
	i1 = './1.png'
	# for i in imgFileList:
	print(i)
	img = cv2.imread(i,0)
	opImg = sketch(img)
	cv2.imshow(i,opImg) # showing the image
	cv2.imwrite(i1,opImg) # save the image

	cv2.waitKey()
	cv2.destroyAllWindows()

