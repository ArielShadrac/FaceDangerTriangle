import numpy as np
import cv2 as cv
import mediapipe as mp
from handtracker import HandDetector
from rich.console import Console
from rich import print

console = Console()

handtxt = (30,30)


class FaceDangerDetector:
    def __init__(self, staticMode=False, maxfaces=2, minDetectionCon=0.5, minTrackCon=0.5):
        self.staticMode = staticMode
        self.maxfaces = maxfaces
        self.minDetectionCon = minDetectionCon
        self.minTrackCon = minTrackCon
        self.mpDraw = mp.solutions.drawing_utils
        self.mpFaceMesh = mp.solutions.face_mesh
        self.faceMesh = self.mpFaceMesh.FaceMesh(static_image_mode=self.staticMode,
                                                max_num_faces=self.maxfaces,
                                                min_detection_confidence=self.minDetectionCon,
                                                min_tracking_confidence=self.minTrackCon)
        self.drawSpec = self.mpDraw.DrawingSpec(thickness=1, circle_radius=2)

# FaceDangerTriangle function

    def FaceDangerTriangle(self, img, draw=True, color= (0,255,0,10)):
        FaceTriangle = [8, 57, 287]
        imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        self.results = self.faceMesh.process(imgRGB)
        h, w = img.shape[:2]

        if self.results.multi_face_landmarks:
            mesh_points = np.array([np.multiply([p.x, p.y], [w, h]).astype(int) for p in
                                    self.results.multi_face_landmarks[0].landmark])
            if draw:
                # Draw triangle, and reducing opacity
                alpha = 0.67
                overlay = img.copy()
                img = cv.fillPoly(img, [mesh_points[FaceTriangle]], color)
                # cv.polylines(img, [mesh_points[FaceTriangle]], color)
                # img2 = cv.polylines(img, [mesh_points[FaceTriangle ]], True, color, 1, cv.LINE_AA)
                cv.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

        return img
    
    # def FaceDangerTriangleEdge(self, img, draw=True, color= (0,255,0,10)):
    #     FaceTriangle = [8, 57, 287]
    #     imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    #     self.results = self.faceMesh.process(imgRGB)
    #     h, w = img.shape[:2]
    #     if self.results.multi_face_landmarks:
    #         mesh_points = np.array([np.multiply([p.x, p.y], [w, h]).astype(int) for p in
    #                                 self.results.multi_face_landmarks[0].landmark])
    #         if draw:
    #             cv.polylines(img, [mesh_points[FaceTriangle ]], True, color, 1, cv.LINE_AA)
    #     return img


# Initializing Pointer class
class HandPointer:
    def __init__(self, color=(0, 255, 0)):
        self.color = color

    def drawPointer(self, img, lmList):
        cv.circle(img, (lmList[8][0], lmList[8][1]), 10, self.color, cv.FILLED)


def main():
    cap = cv.VideoCapture(0)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, 720)  # Set width to 1280 pixels
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, 480)
    detector = FaceDangerDetector(staticMode=False, maxfaces=2, minDetectionCon=0.5, minTrackCon=0.5)
    Hand = HandDetector()
    pointer = HandPointer()  # Crée une instance de HandPointer

    lmList = []  # Initialisation de lmList en dehors de la boucle while
    
    while True:
        success, img = cap.read()
        hands, img = Hand.GetHands(img, draw=True, box=False, corner=True)
        # Vérifier si des mains sont détectées
        if hands:   
            lmList = hands[0]['lmList']
            # Utiliser la méthode drawPointer de HandPointer pour dessiner le pointeur
            pointer.drawPointer(img, lmList)
            img = detector.FaceDangerTriangle(img, draw=True, color=(0, 0, 255, 150))

            # console.print(':warning:','Potential Danger detected!!',  style="red")
            print('[bold red]:warning:[/bold red]', "[red]Potential Danger detected[red], [yellow]Don't enter in the[/yellow] [bold red]red zone[/bold red]")
            cv.putText(img,'Potential Danger detected', handtxt, cv.FONT_HERSHEY_DUPLEX, 0.7,(0,0,255),1)
        else:
            img = detector.FaceDangerTriangle(img, draw=True)
            console.print(':green_circle: Safe!!, No Danger detected!!',  style="green")
            cv.putText(img,'Safe!!, No Danger detected!!', handtxt, cv.FONT_HERSHEY_DUPLEX, 0.7,(0,255,0),1)

        cv.imshow("Image", img)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()



if __name__ == "__main__":
    main()