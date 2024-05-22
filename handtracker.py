import math
import cv2 as cv
import mediapipe as mp


class HandDetector:
    
    def __init__(self, staticMode=False, maxHands=2, modelComplexity=1, detectionCon=0.5, minTrackCon=0.5):
        
        self.staticMode = staticMode
        self.maxHands = maxHands
        self.modelComplexity = modelComplexity
        self.detectionCon = detectionCon
        self.minTrackCon = minTrackCon
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(
                        max_num_hands=self.maxHands,
                        model_complexity=modelComplexity,
                        min_detection_confidence=self.detectionCon,
                        min_tracking_confidence=self.minTrackCon)
        self.mpDraw =  mp.solutions.drawing_utils


    def GetHands(self, img, draw=True, corner=True, box=True, landmarks=True, flipType=True, color=(255,255,255)):

        imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        allHands = []
        h, w, c = img.shape
        if self.results.multi_hand_landmarks:
            for handType, handLms in zip(self.results.multi_handedness, self.results.multi_hand_landmarks):
                myHand = {}
                ## lmList
                mylmList = []
                xList = []
                yList = []
                for id, lm in enumerate(handLms.landmark):
                    px, py, pz = int(lm.x * w), int(lm.y * h), int(lm.z * w)
                    mylmList.append([px, py, pz])
                    xList.append(px)
                    yList.append(py)

                ## bbox
                xmin, xmax = min(xList), max(xList)
                ymin, ymax = min(yList), max(yList)
                boxW, boxH = xmax - xmin, ymax - ymin
                bbox = xmin, ymin, boxW, boxH
                cx, cy = bbox[0] + (bbox[2] // 2),bbox[1] + (bbox[3] // 2)

                myHand["lmList"] = mylmList
                myHand["bbox"] = bbox
                myHand["center"] = (cx, cy)

                if flipType:
                    if handType.classification[0].label == "Right":
                        myHand["type"] = "Left"
                    else:
                        myHand["type"] = "Right"
                else:
                    myHand["type"] = handType.classification[0].label
                allHands.append(myHand)

                if draw:
                    if landmarks:
                        self.mpDraw.draw_landmarks(img, handLms,
                                                    self.mpHands.HAND_CONNECTIONS)
                    if box:
                        cv.rectangle(img, (bbox[0] - 20, bbox[1] - 20),
                                    (bbox[0] + bbox[2] + 20, bbox[1] + bbox[3] + 20),
                                    color, int(1.8))

                    if corner:
                        img = self.Corner(img, bbox)

        return allHands, img
    
    def findDistance(self, p1, p2, img=None):

        x1, y1 = p1
        x2, y2 = p2
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        length = math.hypot(x2 - x1, y2 - y1)
        info = (x1, y1, x2, y2, cx, cy)
        if img is not None:
            cv.circle(img, (x1, y1), 15, (255, 0, 255), cv.FILLED)
            cv.circle(img, (x2, y2), 15, (255, 0, 255), cv.FILLED)
            cv.line(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
            cv.circle(img, (cx, cy), 15, (255, 0, 255), cv.FILLED)
            return length, info, img
        else:
            return length, info
    
    def Corner(self, img, bbox, l=20, t=3, corner_color=(0,0,255)):

        x = bbox[0] - 20
        y = bbox[1] - 20
        x1 = x+bbox[2] + 40
        y1 = y+bbox[3] + 40

# TOP LEFT
        cv.line(img, (bbox[0] - 20,y), (x+l,y),corner_color,t)
        cv.line(img, (x,y), (x,y+l), corner_color,t)
# TOP RIGHT
        cv.line(img, (x1,y), (x1-l,y), corner_color,t)
        cv.line(img, (x1,y), (x1,y+l), corner_color,t)
# BOTTOM LEFT
        cv.line(img, (x, y1), (x + l, y1), corner_color,t)
        cv.line(img, (x, y1), (x, y1 - l), corner_color,t)
# BOTTOM RIGHT
        cv.line(img, (x1, y1), (x1 - l, y1), corner_color,t)
        cv.line(img, (x1, y1), (x1,y1-l), corner_color,t)

        return img

def main():

    cap = cv.VideoCapture(0)
    detector = HandDetector(staticMode=False, maxHands=2, modelComplexity=1, detectionCon=0.5, minTrackCon=0.5)
    
    while True:
        success, img = cap.read()
        hands, img = detector.GetHands(img, draw=True, flipType=True, corner=True, box=True)
        
        
        cv.imshow('Image', img)
        cv.waitKey(1)

if __name__ == "__main__":
    main()