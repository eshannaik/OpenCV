import cv2
import argparse
import numpy as np 
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument("-i","--image",required=True,help="Image path")
args = vars(ap.parse_args())
image_path = args["image"]
img = cv2.imread(image_path)
img = cv2.resize(img,(1024,720))

cTime=0
pTime=0

# while 1:
img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
p = img.reshape((-1,3))
p = np.float32(p)
# print(p)

#stopping criteria 
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,100,0.2) #eps -> reach accuracy needed (epsilon), max_iter -> max iterations

k=8
_,labels,(centers) = cv2.kmeans(p,k,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)

#all pixels to color of centroid
centers = np.uint8(centers)

#flatten
labels = labels.flatten()

segment_image = centers[labels]
segment_image = segment_image.reshape(img.shape) #reshape it back to original shape

#disable a particular cluster to view the other clusters
masked_image = np.copy(img)
masked_image = masked_image.reshape((-1,3))
clusters=6
masked_image[labels==clusters] = [0,0,0]
masked_image = masked_image.reshape(img.shape)

label = f'Cluster {clusters} disabled : Color Black'
cv2.putText(masked_image,str(label),(5,25),cv2.FONT_HERSHEY_COMPLEX,0.5,(0,0,255),2)

#disable all clusters except the one you want to see
m_image = np.copy(img)
m_image = m_image.reshape((-1,3))
clusters=7
m_image[labels!=clusters] = [0,0,0]
m_image = m_image.reshape(img.shape)

label = f'Cluster {clusters} enabled'
cv2.putText(m_image,str(label),(5,25),cv2.FONT_HERSHEY_COMPLEX,0.5,(0,0,255),2)

# cv2.imshow("Image",segment_image)
# cv2.imshow("Image",masked_image)
cv2.imshow("Image",m_image)
# cv2.imwrite("./segemented_image.png",segment_image)
# cv2.imwrite("./masked_image.png",masked_image)
cv2.imwrite("./m_image.png",m_image)
cv2.waitKey(0)
		
if cv2.waitKey(20) & 0xFF == 27 or cv2.getWindowProperty("Image",0) == -1:
	cv2.destroyAllWindows()

# cv2.destroyAllWindows()