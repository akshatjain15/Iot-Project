from imutils.video import VideoStream
import cv2
import imutils
import random
import argparse
import time
import os
import datetime
import urllib.request as ur
import numpy as np

def pred_face():
	while True:
	# to have a maximum width of 400 pixels
		# frame = vs.read()
		try:
			imgResp=ur.urlopen(url)
		except:
			print("ERROR \n Connection not established")
			break
		imgNp=np.array(bytearray(imgResp.read()),dtype=np.uint8)
		image = cv2.imdecode(imgNp,-1)	
		frame = imutils.resize(image, width=400)

		# grab the frame dimensions and convert it to a blob
		(h, w) = frame.shape[:2]
		blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,(300, 300), (104.0, 177.0, 123.0))
		# pass the blob through the network and obtain the detections and
		# predictions
		net.setInput(blob)
		detections = net.forward()

		# loop over the detections
		for i in range(0, detections.shape[2]):
			# extract the confidence (i.e., probability) associated with the
			# prediction
			confidence = detections[0, 0, i, 2]

			# filter out weak detections by ensuring the `confidence` is
			# greater than the minimum confidence
			if confidence < args["confidence"]:
				continue
			return(frame)


file = open('url_read.txt', 'r')
url = file.read() # This value will change according to the link in the application



i = 1
y=10000
# vs = VideoStream(src=0).start()
time.sleep(2.0)

#################################################################################
while(True):
	print("""PRINT 'Q' TO STOP CAMARA""")
	#video_capture = cv2.VideoCapture(0)

	ap = argparse.ArgumentParser()
	# ap.add_argument("-p", "--prototxt", required=True,
	# 	help="path to Caffe 'deploy' prototxt file")
	# ap.add_argument("-m", "--model", required=True,
	# 	help="path to Caffe pre-trained model")
	ap.add_argument("-c", "--confidence", type=float, default=0.5,
		help="minimum probability to filter weak detections")
	try:
		imgResp=ur.urlopen(url)
	except:
		print("ERROR \n Connection not established")
		break
	imgNp=np.array(bytearray(imgResp.read()),dtype=np.uint8)
	image = cv2.imdecode(imgNp,-1)


	args = vars(ap.parse_args())
	# frame = vs.read()
	frame = imutils.resize(image, width=400)
	# grab the frame dimensions and convert it to a blob
	(h, w) = frame.shape[:2]
	net = cv2.dnn.readNetFromCaffe('deploy.prototxt.txt', 'res10_300x300_ssd_iter_140000.caffemodel')

	if(pred_face().any()):
		# frame = vs.read()
		try:
			imgResp=ur.urlopen(url)
		except:
			print("ERROR \n Connection not established")
			break
		imgNp=np.array(bytearray(imgResp.read()),dtype=np.uint8)
		image = cv2.imdecode(imgNp,-1)
		ran = str(random.randint(i,y))
		i = i + 5
		y = y - 5
		current_date = datetime.datetime.today().strftime('%d-%m-%Y')
		newpath = "/home/pi/Desktop/IOT FINAL/DATA/test/"+str(current_date)+""

		if not os.path.exists(newpath):
			os.makedirs(newpath)
		path = newpath +"/"+ ran+".png"
		cv2.imwrite(path, image)

		if cv2.waitKey(33) == ord('q'):
			break

################################################################################3######33
