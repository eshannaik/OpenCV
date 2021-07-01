import cv2
from pygame import mixer
import numpy as np
import math

import object_detection as od

# mixer.init()
# sound = mixer.Sound("./alarm_beep.mp3")

class object_distance():
	focalLength = 0.0  # Focal length 
	angle = 0.0  # Angle 
	fitted_height = 0.0  # Fitted height of the largest contour
	fitted_width = 0.0  # Fitted width of the largest contour
	Width = 1.0  # Known Width of object at certain distance in units
	Height = 1.0  # Known height of object at certain distance in units
	
	def __init__(self,width,height,pixel_height,kDistance):
		self.width=width
		self.height=height

		self.focal_length= (pixel_height*kDistance)/height

	def distance(self,x1,x2,y1,y2):
		dist = math.sqrt((x2-x1)**2 + (y2-y1)**2)
		return dist

	def getDistance(self):
		if (self.fitted_height>0):
			return round(((self.height * self.focal_length)/(self.fitted_height)),2)
		else:
			return 0

	def getAngle(self):
		a = self.width/self.height
		w2 = self.fitted_height * a

		if(self.width != 0 and w2>self.fitted_height):
			return ((1-self.fitted_width)/w2) * 90
		else:
			return 0

	def getFocalLength(self):
		return self.focal_length

	def getFittedBox(self):
		return self.fitted_width,self.fitted_height
