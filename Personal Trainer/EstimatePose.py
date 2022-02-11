#for image processing 
import cv2
from cv2 import WINDOW_NORMAL

#for pose estimation
import mediapipe as mp  

import time

mpDraw = mp.solutions.drawing_utils

#Create the object for pose estimation.
mpPose = mp.solutions.pose
pose = mpPose.Pose()

#read video 
cap = cv2.VideoCapture("Videos/1.mp4")

previous_time = 0 

while True:
  success, img = cap.read()
  imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) #convert the image from BGR to RGB so that we can give it to the object for pose estimation.
  results = pose.process(imgRGB) #give the image to the object for pose estimation.   
  if results.pose_landmarks:
    mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
    for id, lm in enumerate(results.pose_landmarks.landmark):
      h, w, c = img.shape
      cx, cy = int(lm.x*w), int(lm.y*h)
      cv2.circle(img, (cx,cy), 5, (255,0,0), cv2.FILLED)

  current_time = time.time()
  fps = 1/(current_time - previous_time)
  previous_time = current_time

  cv2.putText(img, str(int(fps)), (70,50), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,0), 3)
  cv2.namedWindow("Pushup Evaluation", WINDOW_NORMAL)
  cv2.imshow("Pushup Evaluation", img)
  cv2.waitKey(1) 

