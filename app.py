from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
from pptx import Presentation
from PIL import Image
from cvzone.HandTrackingModule import HandDetector
import cv2
from spire.presentation.common import *
from spire.presentation import *
import subprocess
import shutil
import numpy as np
import os

app = Flask(__name__)

# Function to convert PowerPoint slides to images

def ppt_to_images(pptx_file, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Load the PowerPoint presentation
    presentation = Presentation()
    presentation.LoadFromFile(pptx_file)

    image_files = []
    # Loop through the slides in the presentation
    for i, slide in enumerate(presentation.Slides):
        # Specify the output file name
        file_name = f"{output_folder}/slide_{i}.png"
        # Save each slide as a PNG image
        image = slide.SaveAsImage()
        image.Save(file_name)
        image.Dispose()
        image_files.append(file_name)

    presentation.Dispose()

    return image_files

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Check if the POST request has the file part
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file:
            # Check if the Presentation folder exists, create it if it doesn't
            presentation_folder = "Presentation"
            if not os.path.exists(presentation_folder):
                os.makedirs(presentation_folder)
            
            # Save the uploaded PowerPoint file to the Presentation folder
            pptx_filename = secure_filename(file.filename)
            pptx_path = os.path.join("Presentation", pptx_filename)
            file.save(pptx_path)
            
            # Convert PowerPoint to images
            output_folder = os.path.join("Presentation", "slides")
            ppt_to_images(pptx_path, output_folder)

            # Redirect to the presentation route
            return redirect(url_for("presentation"))
    return render_template("index.html")

@app.route("/presentation")
def presentation():
    width, height = 1280, 720
    gestureThreshold = 300
    folderPath = "Presentation/slides"
    # Camera Setup
    cap = cv2.VideoCapture(0)
    cap.set(3, width)
    cap.set(4, height)

    # # Hand Detector
    detectorHand = HandDetector(detectionCon=0.8, maxHands=1)

    # Variables
    # imgList = []
    delay = 30
    buttonPressed = False
    counter = 0
    # drawMode = False
    imgNumber = 0
    # delayCounter = 0
    annotations = [[]]
    annotationNumber = -1
    annotationStart = False
    hs, ws = int(120 * 1), int(213 * 1)  # width and height of small image

    # Get list of presentation images
    pathImages = sorted(os.listdir(folderPath), key=len)

    while True:
        # Get image frame
        success, img = cap.read()
        img = cv2.flip(img, 1)
        pathFullImage = os.path.join(folderPath, pathImages[imgNumber])
        imgCurrent = cv2.imread(pathFullImage)

        # Find the hand and its landmarks
        hands, img = detectorHand.findHands(img, flipType=True)  # with draw
        # Draw Gesture Threshold line
        cv2.line(img, (0, gestureThreshold), (width, gestureThreshold), (0, 255, 0), 10)

        if hands and buttonPressed is False:  # If hand is detected
            hand = hands[0]
            cx, cy = hand["center"]
            lmList = hand["lmList"]  # List of 21 Landmark points
            fingers = detectorHand.fingersUp(hand)  # List of which fingers are up

            # Constrain values for easier drawing
            xVal = int(np.interp(lmList[8][0], [width // 2, width], [0, width]))
            yVal = int(np.interp(lmList[8][1], [150, height-150], [0, height]))
            indexFinger = lmList[8][0], lmList[8][1]

            if cy <= gestureThreshold:  # If hand is at the height of the face
                if fingers == [1, 0, 0, 0, 0]:
                    annotationStart = False
                    print("Left")

                    if imgNumber > 0:
                        buttonPressed = True
                        imgNumber -= 1
                        annotations = [[]]
                        annotationNumber = -1

                if fingers == [0, 0, 0, 0, 1]:
                    annotationStart = False
                    print("Right")

                    if imgNumber < len(pathImages) - 1:
                        imgNumber += 1
                        buttonPressed = True
                        annotations = [[]]
                        annotationNumber = -1

            if fingers == [0, 1, 1, 0, 0]:
                cv2.circle(imgCurrent, indexFinger, 12, (0, 0, 255), cv2.FILLED)

            if fingers == [0, 1, 0, 0, 0]:
                if annotationStart is False:
                    annotationStart = True
                    annotationNumber += 1
                    annotations.append([])
                annotations[annotationNumber].append(indexFinger)
                cv2.circle(imgCurrent, indexFinger, 12, (0, 0, 255), cv2.FILLED)

            else:
                annotationStart = False

            if fingers == [0, 1, 1, 1, 0]:
                if annotations:
                    annotations.pop(-1)
                    annotationNumber -= 1
                    buttonPressed = True

        else:
            annotationStart = False

        if buttonPressed:
            counter += 1
            if counter > delay:
                counter = 0
                buttonPressed = False

        for i in range(len(annotations)):
            for j in range(len(annotations[i])):
                if j != 0:
                    cv2.line(imgCurrent, annotations[i][j-1], annotations[i][j], (0,0,200), 12)

        imgSmall = cv2.resize(img, (ws, hs))
        h, w, _ = imgCurrent.shape
        imgCurrent[0:hs, w - ws: w] = imgSmall

        cv2.imshow("Slides", imgCurrent)
        cv2.imshow("Image", img)

        key = cv2.waitKey(1)
        if key == ord('q'):
            # shutil.rmtree("Presentation")
            break

    cap.release()
    cv2.destroyAllWindows()

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
