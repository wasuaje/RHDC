#!/usr/bin/python
# face_detect.py
# Face Detection using OpenCV. Based on sample code from:
# http://python.pastebin.com/m76db1d6b
# Usage: python face_detect.py <image_file>

import sys, os
from opencv.cv import *
from opencv.highgui import *
import Image, ImageDraw

FACES='/usr/share/doc/opencv-doc/examples/haarcascades/haarcascade_frontalface_alt.xml'
BODYES='/usr/share/doc/opencv-doc/examples/haarcascades/haarcascade_fullbody.xml'

def detectObjects(image):
  """Converts an image to grayscale and prints the locations of any 
     faces found"""

  grayscale = cvCreateImage(cvSize(image.width, image.height), 8, 1)
  cvCvtColor(image, grayscale, CV_BGR2GRAY)

  storage = cvCreateMemStorage(0)
  cvClearMemStorage(storage)
  cvEqualizeHist(grayscale, grayscale)
  cascade = cvLoadHaarClassifierCascade(FACES,cvSize(1,1))
  faces = cvHaarDetectObjects(grayscale, cascade, storage, 1.2, 2,CV_HAAR_DO_CANNY_PRUNING, cvSize(50,50))

  if faces.total > 0:
    im=image_ctl('open')
    for f in faces:
      x1,y1,x2,y2=f.x,f.y,f.x+f.width,f.y+f.height
      print("[(%d,%d) -> (%d,%d)]" % (x1,y1,x2,y2))
      im=print_rectangle(x1,y1,x2,y2,im)	      #call to a python pil
    image_ctl('save',im)			      #save when all rectangles set

def print_rectangle(x1,y1,x2,y2,im):		#function to draw rectangles
	draw = ImageDraw.Draw(im)
	draw.rectangle([x1,y1,x2,y2],outline="red")
	return im

def image_ctl(action,im=False,filename=sys.argv[1]):
	filename=sys.argv[1].split(".")	
	if action=='open':
		im = Image.open(filename[0]+'.'+filename[1])
		im.save(filename[0]+"_mrk."+filename[1])		#make a copy to work with
	if action=='save':
		im.save(filename[0]+"_mrk."+filename[1])		#ran when finished framing		
	return im

def main():
  image = cvLoadImage(sys.argv[1]);
  detectObjects(image)

if __name__ == "__main__":
  main()

