import cv2
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-i","--image",help="Image Path")
ap.add_argument("-c", "--confidence", type=float, default=0.5,help="minimum probability to filter weak detections, IoU threshold")
args = vars(ap.parse_args())
img_path = args['image']

#Loading the Mask-RCNN model
net = cv2.dnn.readNetFromTensorflow("./dnn/frozen_inference_graph_coco.pb",
								   "./dnn/mask_rcnn_inception_v2_coco_2018_01_28.pbtxt")

#Colors to fill the mask
colors = np.random.randint(0,255,(80,3)) # 80 because this model can detect 80 different objects and 3 because 3 channels

#Getting the image
img = cv2.imread(img_path)
#input image from code
# img = cv2.imread("./source code/road.jpg")
height,width,_ = img.shape
new_img = np.zeros((height,width,3),np.uint8)#getting separate image to show masks of img
# print(img.shape)

#Detect Objects
blob = cv2.dnn.blobFromImage(img,size=(1024,720),swapRB=True) # preprocessing the image
net.setInput(blob)
boxes,masks = net.forward(["detection_out_final","detection_masks"]) #getting final output boxes and masks
detection_count = boxes.shape[2] # getting number of detected objects
# print(detection_count,boxes.shape)

# box
for i in range(detection_count): #looping through all the detections
	box = boxes[0,0,i]
	class_id = box[1] #getting the class type
	score = box[2]

	if score < args['confidence']: #if low confidence then dont show that object
		continue

	#Getting box coordinates
	x = int(box[3] * width) #first coordinate of rectangle is at position 3 
	y = int(box[4] * height) #second corner of box
	x2 = int(box[5] * width) #third corner of box
	y2 = int(box[6] * height) #fourth corner of box

	color = colors[int(class_id)] # a particular color for each object

	#drawing 
	cv2.rectangle(img,(x,y),(x2,y2),(int(color[0]),int(color[1]),int(color[2])),3)
	# cv2.putText(img,f'Confidence : {str(round(score,2))}',(x,y-10),cv2.FONT_HERSHEY_COMPLEX,0.5,(255,255,0),1)
	print("Class Id : ", int(class_id),"Confidence : ", str(round(score,2)))

	roi = new_img[y:y2,x:x2] #getting objects 1 by 1
	roi_height,roi_width,_ = roi.shape

	m = masks[i,int(class_id)]
	# print(m)
	m = cv2.resize(m,(roi_width,roi_height))
	_, m = cv2.threshold(m,0.6,255,cv2.THRESH_BINARY)

	#getting the mask coordinates to draw polygon around the ojects
	c,_ = cv2.findContours(np.array(m,np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE) 
	for cnt in c:
		#Filling the mask(poly) with colors so show segments in the image
		cv2.fillPoly(roi,[cnt],(int(color[0]),int(color[1]),int(color[2])))
	# cv2.imshow("Mask",m)
	# cv2.imshow("ROI",roi)
	# cv2.waitKey(0)



img = cv2.resize(img,(1024,720))
new_img = cv2.resize(new_img,(1024,720))
cv2.imshow("Image",img)
# cv2.imshow("Black Image",new_img)
cv2.waitKey(0)