import cv2
import mediapipe as mp
import numpy as np
from pycaw.pycaw import AudioUtilities

device = AudioUtilities.GetSpeakers()
volume = device.EndpointVolume
minVol, maxVol, _ = volume.GetVolumeRange()
INDEX_FINGER_POINT_INDEX = 8
THUMB_POINT_INDEX = 4


videoCap = cv2.VideoCapture(0)
#Set max and min to unrealistic values so that calibration begins immediately
maxDiff = 0
minDiff = 1

handSolution = mp.solutions.hands
hands = handSolution.Hands()

while True:
        success, img = videoCap.read()
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if success:
            recHands = hands.process(img_rgb)
            if recHands.multi_hand_landmarks:
                for hand in recHands.multi_hand_landmarks:
                    
                    for jointID, point in enumerate(hand.landmark):
                        height, width, channels = img.shape
                        x, y = int(point.x * width), int(point.y * height)
                        cv2.circle(img, (x, y), 10, (255, 255, 255), cv2.FILLED)

                    
                    thumbCoord = np.array((hand.landmark[THUMB_POINT_INDEX].x, hand.landmark[THUMB_POINT_INDEX].y))
                    indexCoord = np.array((hand.landmark[INDEX_FINGER_POINT_INDEX].x, hand.landmark[INDEX_FINGER_POINT_INDEX].y))
                    
                    #Euclidean distance from index to thumb tip
                    distance = np.linalg.norm(thumbCoord - indexCoord)
                   
                    #Distance calibration
                    minDiff = np.minimum(distance, minDiff)
                    maxDiff = np.maximum(distance, maxDiff)
                    #Interpolated Volume, mapped from -65DB to 0
                    normalisedVol = np.interp(distance, [minDiff, maxDiff], [minVol, maxVol])
                    volume.SetMasterVolumeLevel(normalisedVol, None)
                    print(normalisedVol) 
                    
            
            cv2.imshow("CamOutput", img)
            cv2.waitKey(1)
            