import numpy as numpy
import cv2
import argparse
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument('-i','--image',required=True,help="Image Path")
args= vars(ap.parse_args())
img_path = args['image']
img = cv2.imread(img_path)

clicked = False
r=g=b=xpos=ypos=0

c = ["color",'color_name','hex','R','G','B']
df = pd.read_csv('./colors.csv',names=c,header=None)

def draw_function(event,x,y,param,flags):
	if event == cv2.EVENT_LBUTTONDBLCLK:
		global xpos,ypos,r,g,b,clicked
		clicked = True
		xpos=x
		ypos=y
		b,g,r = img[y,x]
		b = int(b)
		g = int(g)
		r = int(r)

cv2.namedWindow('Image')
cv2.setMouseCallback('Image',draw_function)

def colorName(R,G,B):
	minimum=10000
	for i in range(len(df)):
		d = abs(R-int(df.loc[i,"R"])) + abs(G-int(df.loc[i,"G"])) + abs (B-int(df.loc[i,"B"]))
		if(d<=minimum):
			minimum = d
			cname = df.loc[i,"color_name"]
	return cname

while(1):
	cv2.imshow('Image',img)

	if(clicked):
		cv2.rectangle(img,(20,20),(900,60),(b,g,r),-1)
		text = colorName(r,g,b) + "-> R = " + str(r) + " ,G = " + str(g) + " ,B = " + str(b)
		cv2.putText(img,text,(50,50),4,0.9,(255,255,255),1,cv2.LINE_AA)

		if(r+g+b>=600):
			cv2.putText(img,text,(50,50),4,0.9,(0,0,0),1,cv2.LINE_AA)

		clicked =False

	if cv2.waitKey(20) & 0xFF == 27 or cv2.getWindowProperty('Image', 0) == -1:
		break
    
cv2.destroyAllWindows()


