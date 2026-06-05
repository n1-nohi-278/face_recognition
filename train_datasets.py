import cv2
import numpy as np
import os

size = 4
haar_file = 'haarcascade_frontalface_default.xml'
datasets = r'C:\Users\yaswanthi\env\Sc\datasets'

print('Initializing Real-Time Face Recognition...')

# Rebuild the names dictionary dynamically
(images, labels, names, id) = ([], [], {}, 0)
for (subdirs, dirs, files) in os.walk(datasets):
    for subdir in dirs:
        names[id] = subdir
        id += 1

# Load the trained model
model = cv2.face.LBPHFaceRecognizer_create()
model.read('trainer.xml')

# Load the Haar Cascade classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + haar_file)

# --- CAMERA INITIALIZATION OVERHAUL ---
# Trying Index 0 with DirectShow backend first (Best for standard Windows webcams)
webcam = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not webcam.isOpened():
    print("[WARNING] Index 0 failed. Trying camera Index 1...")
    webcam = cv2.VideoCapture(1, cv2.CAP_DSHOW)

if not webcam.isOpened():
    print("[ERROR] Could not open any webcam stream. Please verify no other app (like Zoom or a previous python terminal) is locking the camera.")
    exit()

print("\n [INFO] Camera active! Press 'Esc' key to quit the program.")

while True:
    ret, im = webcam.read()
    
    # If a frame can't be read, stop the script gracefully instead of freezing
    if not ret:
        print("[ERROR] Camera opened but failed to grab a video frame.")
        break
        
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    for (x, y, w, h) in faces:
        cv2.rectangle(im, (x, y), (x + w, y + h), (255, 0, 0), 2)
        face = gray[y:y + h, x:x + w]
        face_resize = cv2.resize(face, (130, 100))
        
        prediction = model.predict(face_resize)
        
        if prediction[1] < 100:
            name = names[prediction[0]]
            confidence = round(100 - prediction[1])
            text_color = (0, 255, 0) 
            label_text = f"{name} - {confidence}%"
        else:
            label_text = "Unknown"
            text_color = (0, 0, 255) 
            
        cv2.putText(im, label_text, (x, y - 10), cv2.FONT_HERSHEY_PLAIN, 1.5, text_color, 2)
        
    # Open the visual GUI window
    cv2.imshow('Face Recognition System', im)
    
    # Wait for key press
    key = cv2.waitKey(10)
    if key == 27:
        break

webcam.release()
cv2.destroyAllWindows()
print("\n [INFO] System shutdown.")