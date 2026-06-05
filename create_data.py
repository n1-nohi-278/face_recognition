import cv2
import os

# Define the XML file for face detection
haar_file = 'haarcascade_frontalface_default.xml'

# Master folder name and sub-folder name for the individual user
datasets = 'datasets'
sub_data = 'smriti mandhana' 
 # Change this to your name when you run it!

# Combine paths: datasets/sanjay
path = os.path.join(datasets, sub_data)

# If the folder directory doesn't exist, create it automatically
if not os.path.isdir(path):
    os.makedirs(path)

# Define the standard size to resize the cropped faces
(width, height) = (130, 100)

# Load the Haar Cascade classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + haar_file)

# Start webcam capture (0 is default, use 1 if you have an external webcam like him)
webcam = cv2.VideoCapture(0)

print(f"\n [INFO] Initializing camera. Capturing images for '{sub_data}'...")
count = 1

while count < 31:
    print(count)
    (_, im) = webcam.read()
    
    # Convert image to grayscale
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.3, 4)
    
    for (x, y, w, h) in faces:
        # Draw rectangle around the face
        cv2.rectangle(im, (x, y), (x + w, y + h), (255, 0, 0), 2)
        
        # Crop the face from the frame
        face = gray[y:y + h, x:x + w]
        
        # Resize the face image to standard dimensions (130x100)
        face_resize = cv2.resize(face, (width, height))
        
        # Save the image into the datasets/sanjay/ directory
        cv2.imwrite('%s/%s.png' % (path, count), face_resize)
        
        count += 1
        
    # Display the window
    cv2.imshow('OpenCV', im)
    
    # Clear window loop if 'esc' key is pressed
    key = cv2.waitKey(10)
    if key == 27:
        break

print(f"\n [INFO] Dataset created successfully for {sub_data}!")
webcam.release()
cv2.destroyAllWindows()