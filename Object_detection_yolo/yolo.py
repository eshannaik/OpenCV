import cv2
import matplotlib.pyplot as plt
import numpy as np
import argparse

yolo = cv2.dnn.readNet("./yolov3.weights","./darknet/cfg/yolov3.cfg")

classes = []
p = 0

with open("./darknet/data/coco.names","r") as f:
	classes = f.read().splitlines()

#total number of classes
print(len(classes))

ap = argparse.ArgumentParser()
ap.add_argument("-i","--image",help="Image Path")
ap.add_argument("-c", "--confidence", type=float, default=0.8,help="minimum probability to filter weak detections, IoU threshold")
ap.add_argument("-t", "--threshold", type=float, default=0.3,help="threshold when applying non-maxima suppression")
args = vars(ap.parse_args())
img_path = args['image']
img = cv2.imread(img_path)
(H, W) = img.shape[:2]

new_img = cv2.dnn.blobFromImage(img,1/255,(320,320),(0,0,0),swapRB=True,crop=False) #resize to 320x320 and swapRB -> it reads a BGR to RGB

#printing image shape
print(new_img.shape)

#printing image
i = new_img[0].reshape(320,320,3)
plt.imshow(i)

#input layer
yolo.setInput(new_img)
#output layer
output_layer = yolo.getUnconnectedOutLayersNames()
layeroutput = yolo.forward(output_layer)

#detecting objects
box = []
confidences = []
class_id = []

for o in layeroutput:
	for d in o:
		score = d[5:] #first 4 boxes is responsible for the position of the boxes
		c_id = np.argmax(score)
		c = score[c_id]

		if c > args['confidence']: #threshold
			boxes = d[0:4] * np.array([W, H, W, H])
			(center_x, center_y, width, height) = boxes.astype("int")

			#corner values
			x = int(center_x - width/2)
			y = int(center_y - height/2)

			box.append([x,y,int(width),int(height)])
			confidences.append(float(c))
			class_id.append(c_id)
			


print(len(box))

indexes = cv2.dnn.NMSBoxes(box,confidences,args['confidence'],args['threshold'])

colors = np.random.uniform(0,255,size=(len(box),3))

#adding font to boxes
for i in indexes.flatten():
	x,y,w,h = box[i]

	l = str(classes[class_id[i]])
	confi = str(round(confidences[i],2))
	color = colors[i]
	p=p+1

	cv2.rectangle(img,(x,y),(x+w,y+h),color,1) # 1 is size of rectangle box
	cv2.putText(img,l + " = " + confi,(x,y-5),cv2.FONT_HERSHEY_SIMPLEX,0.5,color,2) #adding text 
	cv2.rectangle(img,(5,20),(400,2),(0,0,0),-1)
	cv2.putText(img,f"The number of objects in the image are:{int(p)}",(10,15),cv2.FONT_HERSHEY_COMPLEX,0.5,(0,255,0),1)

#saving the image
cv2.imwrite("./img.jpg",img)
while 1:
	img = cv2.resize(img,(1324,712))
	cv2.imshow('Image',img)

	if cv2.waitKey(20) & 0xFF==27 or cv2.getWindowProperty('Image',0)==-1:
		break

cv2.destroyAllWindows()



