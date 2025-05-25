# Importation des bibliothèques nécessaires
import numpy as np
import cv2 as cv
import mediapipe as mp
from handtracker import HandDetector  # Module personnalisé pour le suivi de la main
from rich.console import Console
from rich import print

console = Console()

# Position du texte affiché à l'écran
handtxt = (30, 30)

# Classe pour détecter les dangers via les zones critiques du visage
class FaceDangerDetector:
    def __init__(self, staticMode=False, maxfaces=2, minDetectionCon=0.5, minTrackCon=0.5):
        # Initialisation des paramètres de MediaPipe Face Mesh
        self.staticMode = staticMode
        self.maxfaces = maxfaces
        self.minDetectionCon = minDetectionCon
        self.minTrackCon = minTrackCon
        self.mpDraw = mp.solutions.drawing_utils
        self.mpFaceMesh = mp.solutions.face_mesh
        self.faceMesh = self.mpFaceMesh.FaceMesh(
            static_image_mode=self.staticMode,
            max_num_faces=self.maxfaces,
            min_detection_confidence=self.minDetectionCon,
            min_tracking_confidence=self.minTrackCon)
        self.drawSpec = self.mpDraw.DrawingSpec(thickness=1, circle_radius=2)

    # Fonction pour dessiner un triangle sur une zone critique du visage (menton et joues)
    def FaceDangerTriangle(self, img, draw=True, color=(0, 255, 0, 10)):
        FaceTriangle = [8, 57, 287]  # Indices des points du triangle (menton et joues)
        imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        self.results = self.faceMesh.process(imgRGB)
        h, w = img.shape[:2]

        if self.results.multi_face_landmarks:
            # Récupérer les coordonnées des points du visage détecté
            mesh_points = np.array([np.multiply([p.x, p.y], [w, h]).astype(int)
                                    for p in self.results.multi_face_landmarks[0].landmark])
            if draw:
                # Dessiner un triangle semi-transparent sur le visage
                alpha = 0.67
                overlay = img.copy()
                img = cv.fillPoly(img, [mesh_points[FaceTriangle]], color)
                cv.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

        return img

# Classe pour dessiner un pointeur sur la main détectée
class HandPointer:
    def __init__(self, color=(0, 255, 0)):
        self.color = color

    # Dessine un cercle sur le bout de l'index (point 8 dans Mediapipe)
    def drawPointer(self, img, lmList):
        cv.circle(img, (lmList[8][0], lmList[8][1]), 10, self.color, cv.FILLED)

# Fonction principale
def main():
    # Initialisation de la capture vidéo
    cap = cv.VideoCapture(0)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, 720)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, 480)

    # Création des objets nécessaires
    detector = FaceDangerDetector()
    Hand = HandDetector()
    pointer = HandPointer()
    lmList = []  # Liste des landmarks de la main

    while True:
        success, img = cap.read()  # Lire une image de la webcam

        # Détection de la main avec la méthode personnalisée
        hands, img = Hand.GetHands(img, draw=True, box=False, corner=True)

        if hands:
            # Si une main est détectée, récupérer les landmarks
            lmList = hands[0]['lmList']

            # Dessiner le pointeur sur l'index
            pointer.drawPointer(img, lmList)

            # Dessiner le triangle de danger sur le visage en rouge
            img = detector.FaceDangerTriangle(img, draw=True, color=(0, 0, 255, 150))

            # Afficher un message d'avertissement
            print('[bold red]:warning:[/bold red]', "[red]Potential Danger detected[red], [yellow]Don't enter in the[/yellow] [bold red]red zone[/bold red]")
            cv.putText(img, 'Potential Danger detected', handtxt, cv.FONT_HERSHEY_DUPLEX, 0.7, (0, 0, 255), 1)
        else:
            # Si aucune main n'est détectée, juste dessiner le triangle en vert
            img = detector.FaceDangerTriangle(img, draw=True)
            console.print(':green_circle: Safe!!, No Danger detected!!', style="green")
            cv.putText(img, 'Safe!!, No Danger detected!!', handtxt, cv.FONT_HERSHEY_DUPLEX, 0.7, (0, 255, 0), 1)

        # Afficher le résultat
        cv.imshow("Image", img)

        # Quitter la boucle si l'utilisateur appuie sur 'q'
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    # Libérer la caméra et fermer les fenêtres
    cap.release()
    cv.destroyAllWindows()

# Point d'entrée du programme
if __name__ == "__main__":
    main()
