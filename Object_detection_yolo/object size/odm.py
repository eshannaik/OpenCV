import cv2

class HomogeneousDetector():
	def __init__(self):
		pass

	def detect_object(self,img):
		gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

		#19 - size of pixel neighbour used to cal thres value,5-constant substracted from mean of neighbour pixels
		mask = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY_INV,19,5) 

		#input,contour retrieval mode,contour approximation method
		c,_ = cv2.findContours(mask,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

		contours = []

		for cnt in c:
			area = cv2.contourArea(cnt)
			if(area>2000):
				# Find the shape of the contour
				# cnt = cv2.approxPolyDP(cnt, 0.03 *cv2.arcLength(cnt,True),True)
				contours.append(cnt)

		return contours