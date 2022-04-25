from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import time
import cv2
from playsound import playsound
from colorama import Fore, Style


ap = argparse.ArgumentParser()
ap.add_argument("-p", "--prototxt", required=True,
	help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", required=True,
	help="path to Caffe pre-trained model")
ap.add_argument("-c", "--confidence", type=float, default=0.2,
	help="minimum probability to filter weak predictions")
args = vars(ap.parse_args())

CLASSES = ["_", "background", "_", "bird", "_",
           "_", "_", "_", "cat", "_", "cow", "_",
           "dog", "horse", "_", "person", "_", "sheep",
           "_", "_", "_"]

COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

print("[INFO] starting video stream...")
vs = VideoStream(src=1).start()

# warm up the camera for a couple of seconds
#time.sleep(2.0)

fps = FPS().start()

soundCounter = 0
soundCounterSpeed = 45
personVisibleFor = 0
framesElapsedSincePerson = 1000

while True:
	soundCounter += 1
	framesElapsedSincePerson += 1

	if framesElapsedSincePerson >= 100:
		personVisibleFor = 0

	frame = vs.read()
	frame = imutils.resize(frame, width=800)
	#print(frame.shape) # (225, 400, 3)

	(h, w) = frame.shape[:2]

	resized_image = cv2.resize(frame, (300, 300))


	blob = cv2.dnn.blobFromImage(resized_image, (1/127.5), (300, 300), 127.5, swapRB=True)

	net.setInput(blob) # net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

	# Predictions:
	predictions = net.forward()

	for i in np.arange(0, predictions.shape[2]):

		confidence = predictions[0, 0, i, 2]

		if confidence > args["confidence"]:

			idx = int(predictions[0, 0, i, 1])

			box = predictions[0, 0, i, 3:7] * np.array([w, h, w, h])
			
			(startX, startY, endX, endY) = box.astype("int")

			label = CLASSES[idx]

			# alarm system
			if CLASSES[idx] == 'person' and not personVisibleFor >= 75:
				personVisibleFor += 1

				if personVisibleFor == 1 and framesElapsedSincePerson >= 100:
					playsound('beep-02.mp3', False)
					print(Fore.CYAN + "bad beep")
				
				framesElapsedSincePerson = 0


			if CLASSES[idx] == 'person' and soundCounter >= soundCounterSpeed and personVisibleFor >= 25:
				playsound('beep-02.mp3', False)
				print(Fore.GREEN + "good beep")
				soundCounter = 0

			#print(CLASSES[idx], idx)

			if label == "person":
				print(Fore.RED + "PERSON DETECTED")

			# bounding box
			cv2.rectangle(frame, (startX, startY), (endX, endY),
				COLORS[idx], 2)
			y = startY - 15 if startY - 15 > 15 else startY + 15

			cv2.putText(frame, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)
		
	cv2.imshow("Frame", frame)

	key = cv2.waitKey(1) & 0xFF

	if key == ord("q"):
		break

	fps.update()

fps.stop()

print(f"[INFO] Elapsed Time: {round(fps.elapsed(), 3)}")
print(f"[INFO] Approximate FPS: {round(fps.fps(), 2)}")

cv2.destroyAllWindows()

vs.stop()