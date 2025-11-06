import cv2
import mediapipe as mp
import numpy as np
from pycaw.pycaw import AudioUtilities

device = AudioUtilities.GetSpeakers()
volume = device.EndpointVolume
INDEX_FINGER_POINT_INDEX = 8
THUMB_POINT_INDEX = 4


videoCap = cv2.VideoCapture(0)
#Set max and min to unrealistic values so that calibration begins immediately
max_diff = 0
min_diff = 1

handSolution = mp.solutions.hands
hands = handSolution.Hands()

while True:
        success, img = videoCap.read()
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if success:
            recHands = hands.process(img_rgb)
            if recHands.multi_hand_landmarks:
                for hand in recHands.multi_hand_landmarks:
                    
                    for joint_id, point in enumerate(hand.landmark):
                        h, w, c = img.shape
                        x, y = int(point.x * w), int(point.y * h)
                        cv2.circle(img, (x, y), 10, (255, 255, 255), cv2.FILLED)

                    
                    thumb_coord = np.array((hand.landmark[THUMB_POINT_INDEX].x, hand.landmark[THUMB_POINT_INDEX].y))
                    index_coord = np.array((hand.landmark[INDEX_FINGER_POINT_INDEX].x, hand.landmark[INDEX_FINGER_POINT_INDEX].y))
                    
                    #Euclidean distance from index to thumb tip
                    distance = np.linalg.norm(thumb_coord - index_coord)
                   
                    #Distance calibration
                    min_diff = np.minimum(distance, min_diff)
                    max_diff = np.maximum(distance, max_diff)
                    #Interpolated Volume, mapped from -65DB to 0
                    normalisedVol = np.interp(distance, [min_diff, max_diff], [-65.0, 0.0])
                    volume.SetMasterVolumeLevel(normalisedVol, None)
                    print(normalisedVol) 
                    
            
            cv2.imshow("CamOutput", img)
            cv2.waitKey(1)