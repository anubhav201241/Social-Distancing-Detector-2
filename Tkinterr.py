from tkinter import *
from mylib import config, thread
from mylib.detection import detect_people
import cv2
from imutils.video import VideoStream, FPS
from PIL import Image, ImageTk
from playsound import playsound
from scipy.spatial import distance as dist
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import argparse
import imutils
import time

from playsound import playsound
def Social_distance():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", type=str, default="",
                    help="path to (optional) input video file")
    ap.add_argument("-o", "--output", type=str, default="",
                    help="path to (optional) output video file")
    ap.add_argument("-d", "--display", type=int, default=1,
                    help="whether or not output frame should be displayed")
    args = vars(ap.parse_args())

    labelsPath = "coco.names"
    LABELS = open(labelsPath).read().strip().split("\n")

    weightsPath = 'yolov4-tiny.weights'
    configPath = 'yolov4-tiny.cfg'
    print("[INFO] loading YOLO from disk...")
    net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

    if config.USE_GPU:
        # set CUDA as the preferable backend and target
        print("[INFO] setting preferable backend and target to CUDA...")
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

    ln = net.getLayerNames()
    ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    # if a video path was not supplied, grab a reference to the camera
    if not args.get("input", False):
        print("[INFO] Starting the live stream..")
        vs = cv2.VideoCapture('4.mp4')
        if config.Thread:
            cap = thread.ThreadingClass(config.url)
        time.sleep(2.0)

    #Grab a reference to the video file
    else:
        print("[INFO] Starting the video..")
        vs = cv2.VideoCapture(args["input"])
        if config.Thread:
            cap = thread.ThreadingClass(args["input"])

    writer = None
    # start the FPS counter
    fps = FPS().start()

    # loop over the frames from the video stream
    while True:
        # read the next frame from the file
        if config.Thread:
            frame = cap.read()

        else:
            (grabbed, frame) = vs.read()
            # Frame was not grabbed, then we have reached the end of the stream
            if not grabbed:
                break

        # Resize the frame and detect the people
        cv2.putText(frame, "Social Distance Detector", (130, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)
        frame = imutils.resize(frame, width=700)
        results = detect_people(frame, net, ln,
                                personIdx=LABELS.index("person"))
        # initialize the set of indexes that violate the max/min social distance limits
        serious = set()
        abnormal = set()
     
        if len(results) >= 2:

            centroids = np.array([r[2] for r in results])
            D = dist.cdist(centroids, centroids, metric="euclidean")

            # loop over the upper triangular of the distance matrix
            for i in range(0, D.shape[0]):
                for j in range(i + 1, D.shape[1]):

                    if D[i, j] < config.MIN_DISTANCE:
                        # update our violation set with the indexes of
                        # the centroid pairs
                        serious.add(i)
                        serious.add(j)
                        playsound("beep-01a.wav")

                    # update our abnormal set if the centroid distance is below max distance limit
                    if (D[i, j] < config.MAX_DISTANCE) and not serious:
                        abnormal.add(i)
                        abnormal.add(j)

        # loop over the results
        for (i, (prob, bbox, centroid)) in enumerate(results):

            (startX, startY, endX, endY) = bbox
            (cX, cY) = centroid
            color = (0, 255, 0)

            # if the index pair exists within the violation/abnormal sets, then update the color
            if i in serious:
                text = "Red Alert"
                color = (0, 0, 255)
                cv2.putText(frame, text, (startX, startY - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            elif i in abnormal:
                text = 'Normal'
                cv2.putText(frame, text, (startX, startY - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                color = (0, 255, 0)

            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
        text = "Social Distance Violation: {}".format(len(serious))
        cv2.putText(frame, text, (10, frame.shape[0] - 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.90, (0, 0, 255), 2)

        text1 = "Safe Count: {}".format(len(abnormal))
        cv2.putText(frame, text1, (10, frame.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)


        if args["display"] > 0:
            # show the output frame
            cv2.imshow("Real-Time Monitoring/Analysis Window", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break
        # update the FPS counter
        fps.update()

        if args["output"] != "" and writer is None:
            # initialize our video writer
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            writer = cv2.VideoWriter(args["output"], fourcc, 25,
                                     (frame.shape[1], frame.shape[0]), True)

        if writer is not None:
            writer.write(frame)

    # Display FPS information
    fps.stop()
    print("===========================")
    print("[INFO] Elasped time: {:.2f}".format(fps.elapsed()))
    print("[INFO] Approx. FPS: {:.2f}".format(fps.fps()))
    cv2.destroyAllWindows()

def mask():
    proto_txt_path = 'deploy.prototxt'
    model_path = 'res10_300x300_ssd_iter_140000.caffemodel'
    face_detector = cv2.dnn.readNetFromCaffe(proto_txt_path, model_path)

    mask_detector = load_model('mask_detector.model')

    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        frame = imutils.resize(frame, width=400)
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104, 177, 123))

        face_detector.setInput(blob)
        detections = face_detector.forward()

        faces = []
        bbox = []
        results = []

        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]

            if confidence > 0.5:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                face = frame[startY:endY, startX:endX]
                face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                face = cv2.resize(face, (224, 224))
                face = img_to_array(face)
                face = preprocess_input(face)
                face = np.expand_dims(face, axis=0)

                faces.append(face)
                bbox.append((startX, startY, endX, endY))

        if len(faces) > 0:
            results = mask_detector.predict(faces)

        for (face_box, result) in zip(bbox, results):
            (startX, startY, endX, endY) = face_box
            (mask, withoutMask) = result
            cv2.putText(frame, "Mask Detector", (95, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
            label = ""
            if mask > withoutMask:
                label = "Mask"
                color = (0, 255, 0)
            else:
                label = "No Mask"
                color = (0, 0, 255)
                playsound("beep-01a.wav")

            cv2.putText(frame, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

            cv2.imshow("Frame", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break


root = Tk()
root.title("Project")
root.geometry('800x600')
img_file = Image.open('im2.png')
bg = ImageTk.PhotoImage(img_file)
bgl = Label(root, image=bg)
bgl.place(x=0, y=0)
lbl = Label(root, text="Team vision", font=('Helvetica', 25, 'underline'))
lbl.pack()
bt1 = Button(root, text='Social Distance detector', bd=8, command=Social_distance)
bt1.place(x=600, y=300)
bt2 = Button(root, text="Face mask Detection", bd=8, command=mask)
bt2.place(x=60, y=300)
root.resizable(0, 0)
root.mainloop()
