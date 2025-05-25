# Importation des bibliothèques nécessaires
import math
import cv2 as cv
import mediapipe as mp

# Définition de la classe HandDetector
class HandDetector:

    def __init__(self, staticMode=False, maxHands=2, modelComplexity=1, detectionCon=0.5, minTrackCon=0.5):
        # Paramètres de détection de mains
        self.staticMode = staticMode  # True si on utilise des images fixes
        self.maxHands = maxHands  # Nombre maximum de mains à détecter
        self.modelComplexity = modelComplexity  # Complexité du modèle (1 = standard)
        self.detectionCon = detectionCon  # Seuil de confiance pour la détection
        self.minTrackCon = minTrackCon  # Seuil de confiance pour le tracking

        # Initialisation du modèle MediaPipe Hands
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(
            max_num_hands=self.maxHands,
            model_complexity=modelComplexity,
            min_detection_confidence=self.detectionCon,
            min_tracking_confidence=self.minTrackCon
        )

        self.mpDraw = mp.solutions.drawing_utils  # Pour dessiner les landmarks

    # Méthode principale pour détecter les mains et retourner infos + image annotée
    def GetHands(self, img, draw=True, corner=True, box=True, landmarks=True, flipType=True, color=(255,255,255)):
        imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)  # Conversion en RGB pour MediaPipe
        self.results = self.hands.process(imgRGB)  # Traitement de l'image

        allHands = []  # Liste pour stocker les données de toutes les mains détectées
        h, w, c = img.shape  # Dimensions de l'image

        if self.results.multi_hand_landmarks:
            # Parcourir chaque main détectée
            for handType, handLms in zip(self.results.multi_handedness, self.results.multi_hand_landmarks):
                myHand = {}
                mylmList = []  # Liste des points clés de la main
                xList, yList = [], []

                # Extraire les coordonnées des landmarks
                for id, lm in enumerate(handLms.landmark):
                    px, py, pz = int(lm.x * w), int(lm.y * h), int(lm.z * w)
                    mylmList.append([px, py, pz])
                    xList.append(px)
                    yList.append(py)

                # Calcul de la bounding box autour de la main
                xmin, xmax = min(xList), max(xList)
                ymin, ymax = min(yList), max(yList)
                boxW, boxH = xmax - xmin, ymax - ymin
                bbox = xmin, ymin, boxW, boxH

                # Calcul du centre de la main
                cx, cy = bbox[0] + (bbox[2] // 2), bbox[1] + (bbox[3] // 2)

                # Stockage des données
                myHand["lmList"] = mylmList
                myHand["bbox"] = bbox
                myHand["center"] = (cx, cy)

                # Détection du type de main (droite/gauche inversée si flipType)
                if flipType:
                    if handType.classification[0].label == "Right":
                        myHand["type"] = "Left"
                    else:
                        myHand["type"] = "Right"
                else:
                    myHand["type"] = handType.classification[0].label

                allHands.append(myHand)

                # Dessin sur l'image si demandé
                if draw:
                    if landmarks:
                        self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
                    if box:
                        cv.rectangle(img, (bbox[0] - 20, bbox[1] - 20),
                                     (bbox[0] + bbox[2] + 20, bbox[1] + bbox[3] + 20),
                                     color, int(1.8))
                    if corner:
                        img = self.Corner(img, bbox)

        return allHands, img

    # Calculer la distance entre deux points
    def findDistance(self, p1, p2, img=None):
        x1, y1 = p1
        x2, y2 = p2
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2  # Centre entre les deux points
        length = math.hypot(x2 - x1, y2 - y1)  # Distance euclidienne
        info = (x1, y1, x2, y2, cx, cy)

        if img is not None:
            # Dessiner la distance sur l'image
            cv.circle(img, (x1, y1), 15, (255, 0, 255), cv.FILLED)
            cv.circle(img, (x2, y2), 15, (255, 0, 255), cv.FILLED)
            cv.line(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
            cv.circle(img, (cx, cy), 15, (255, 0, 255), cv.FILLED)
            return length, info, img
        else:
            return length, info

    # Dessin de coins stylisés autour de la bounding box
    def Corner(self, img, bbox, l=20, t=3, corner_color=(0, 0, 255)):
        x = bbox[0] - 20
        y = bbox[1] - 20
        x1 = x + bbox[2] + 40
        y1 = y + bbox[3] + 40

        # Haut gauche
        cv.line(img, (x, y), (x + l, y), corner_color, t)
        cv.line(img, (x, y), (x, y + l), corner_color, t)
        # Haut droit
        cv.line(img, (x1, y), (x1 - l, y), corner_color, t)
        cv.line(img, (x1, y), (x1, y + l), corner_color, t)
        # Bas gauche
        cv.line(img, (x, y1), (x + l, y1), corner_color, t)
        cv.line(img, (x, y1), (x, y1 - l), corner_color, t)
        # Bas droit
        cv.line(img, (x1, y1), (x1 - l, y1), corner_color, t)
        cv.line(img, (x1, y1), (x1, y1 - l), corner_color, t)

        return img

# Fonction principale pour tester la détection
def main():
    cap = cv.VideoCapture(0)  # Ouverture de la webcam
    detector = HandDetector()  # Création de l'instance

    while True:
        success, img = cap.read()  # Lecture image
        hands, img = detector.GetHands(img, draw=True, flipType=True, corner=True, box=True)

        cv.imshow('Image', img)  # Affichage image
        cv.waitKey(1)  # Attente (1 ms) pour boucle continue

# Point d'entrée du programme
if __name__ == "__main__":
    main()
