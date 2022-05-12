# import the necessary packages
from .config import NON_MAXIMA_SUPP_THRESH
from .config import MIN_CONFIDENCE
import numpy as np
import cv2

def detect_people(frame, net, ln, personIdx=0):

	(H, W) = frame.shape[:2]
	results = []

	# Creating blob from input frm and using forward
	blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
	net.setInput(blob)
	layerOutputs = net.forward(ln)

	boxes = []
	centroids = []
	confidences = []

	for output in layerOutputs:
		# loop over each of the detections
		for detection in output:
			# print(detection)
			scores = detection[5:]
			classID = np.argmax(scores)
			# print(classID)
			confidence = scores[classID]

			if classID == personIdx and confidence > MIN_CONFIDENCE: # Checking person ID match & confidence
				# scale the bounding box coordinates back relative to the size of the image, 
				# keeping in mind that YOLO actually returns the center (x, y)-coordinates of
				# the bounding box followed by the boxes' width and height
				print("Class Id:" + str(classID))
				print("Confidence: " + str(confidence))
				box = detection[0:4] * np.array([W, H, W, H])
				(centerX, centerY, width, height) = box.astype("int")

				# use the center (x, y)-coordinates to derive the top and and left corner of the bounding box
				x = int(centerX - (width / 2))
				y = int(centerY - (height / 2))

				# update list of bounding box coordinates, centroids, and confidences
				boxes.append([x, y, int(width), int(height)])
				centroids.append((centerX, centerY))
				confidences.append(float(confidence))

	# Apply NMS to remove redundant BBX
	idxs = cv2.dnn.NMSBoxes(boxes, confidences, MIN_CONFIDENCE, NON_MAXIMA_SUPP_THRESH)

	if len(idxs) > 0:
		for i in idxs.flatten():
			# extract the bounding box coordinates
			(x, y) = (boxes[i][0], boxes[i][1])
			(w, h) = (boxes[i][2], boxes[i][3])

			r = (confidences[i], (x, y, x + w, y + h), centroids[i])
			results.append(r)

	# return the list of results
	return results