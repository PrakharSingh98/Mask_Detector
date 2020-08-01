from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import winsound

def detect_mask(frame,faceNet,maskNet):
    h,w = frame.shape[:2]
    # cv2.dnn.blobFromImage(image,scale,(size,size),(mean-R,mean-G,mean-B)) This mean will be substracted
    blob = cv2.dnn.blobFromImage(frame, 1.0, (224,224), (104.0, 177.0, 123.0))
    faces =[]
    locs = []
    preds = []
    # Pass the image to obtain faces ie. ROI
    faceNet.setInput(blob)
    detections = faceNet.forward()
    for i in range(0,detections.shape[2]):
        conf = detections[0][0][i][2]
        # Check if minimum confidence is met
        if conf > 0.5:
            # Compute x and y co-ordinates of the face
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype('int')
            # Extract ROI from the image
            face = frame[startY:endY, startX:endX]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)
            faces.append(face)
            locs.append((startX, startY, endX, endY))
    if len(faces) > 0:
        # For faster inference we'll make batch predictions on *all*
        # faces at the same time rather than one-by-one predictions using for loop
        faces = np.array(faces, dtype="float32")
        preds = maskNet.predict(faces, batch_size=32)

    # Return a 2-tuple of the face locations and their corresponding prediction
    return (locs, preds)

# load our caffe based face detector model from disk
prototxtPath = r"face_detector\deploy.prototxt" # This contains the architecture of model
weightsPath = r"face_detector\res10_300x300_ssd_iter_140000.caffemodel" # This contains actual weights
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

# load the face mask detector model from disk
maskNet = load_model("mask_detector.model")
print("[INFO] Succesfully loaded both models")

def beep():
    frequency = 2500  # Set Frequency To 2500 Hertz
    duration = 100  # Set Duration To 1000 ms == 1 second
    winsound.Beep(frequency, duration)

# Starting LIVE video
cap = cv2.VideoCapture(0)
while True:
    _,frame = cap.read()
    frame = cv2.resize(frame,(700,700))
    (locs, preds) = detect_mask(frame, faceNet, maskNet)
    # Looping over detected faces 
    for (loc,pred) in zip(locs,preds):
        # Start unpacking the contents of location and predictions
        (startX, startY, endX, endY) = loc
        # Assigns the probability of each class to it's label
        (mask, withoutMask) = pred
        label = "Mask" if mask > withoutMask else "No Mask"
        color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
        label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)
        # Display the label and bounding box on the frame
        cv2.putText(frame, label, (startX, startY - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
        cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
    # show the output frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

# do a bit of cleanup
cv2.destroyAllWindows()
cap.release()
