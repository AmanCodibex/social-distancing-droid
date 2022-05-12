# imports
from helper import config as config
from helper.detection import detect_people
from scipy.spatial import distance as dist
import numpy as np
import argparse
import imutils
import cv2
import os

# input args mandation
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", type=str, default="",
	help="path to (optional) input video file")
args = vars(ap.parse_args())

# COCO & YOLO
labelsPath = os.path.sep.join([config.YOLO_PATH, "coco.names"])
print(labelsPath)
LABELS = open(labelsPath).read().strip().split("\n")

weightsPath = os.path.sep.join([config.YOLO_PATH, "yolov3.weights"])
configPath = os.path.sep.join([config.YOLO_PATH, "yolov3.cfg"])

# YOLO From disk load
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

# Determining Layers that are needed from YOLO
ln = net.getLayerNames()
print(net.getUnconnectedOutLayers())
print(len(ln))
ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]
print(len(ln))

# Video Stream
vs = cv2.VideoCapture(args["input"] if args["input"] else 0)
writer = None

while True:

	(grabbed, frame) = vs.read()

	if not grabbed:
		break

	frame = imutils.resize(frame, width=700) #Resizing
	results = detect_people(frame, net, ln)

	violate = set() #Holds indices that are not following social distancing

	if len(results) >= 2:
		centroids = np.array([r[2] for r in results])
		D = dist.cdist(centroids, centroids, metric="euclidean")

		for i in range(0, D.shape[0]):
			for j in range(i + 1, D.shape[1]):
				# Checking distance between centroid pairs for less than MIN_DISTANCE
				if D[i, j] < config.MIN_DISTANCE:
					#Adding index in violate var
					violate.add(i)
					violate.add(j)

	for (i, (prob, bbox, centroid)) in enumerate(results):
		
		(startX, startY, endX, endY) = bbox
		(cX, cY) = centroid
		# Green colour for regular box
		color = (0, 255, 0)

		# Red colour if it has been violated
		if i in violate:
			color = (0, 0, 255)

		cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2) #Box around the person
		cv2.circle(frame, (cX, cY), 5, color, 1) #Centre of mass of person

	text = "Violations: {}".format(len(violate)) # Showing violations
	cv2.putText(frame, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 0, 255), 3) #Putting txt on frame

	cv2.imshow("Covid Violation Test", frame)
	key = cv2.waitKey(1) & 0xFF

	# Exit on q key press
	if key == ord("q"):
		break

	if writer is not None:
		writer.write(frame)
