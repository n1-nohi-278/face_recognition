import cv2
import os

# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Path to trained model and cascade classifier
trainer_path = os.path.join(script_dir, 'trainer.xml')
haar_file = 'haarcascade_frontalface_default.xml'
haar_path = os.path.join(script_dir, haar_file)
if not os.path.exists(haar_path):
    haar_path = cv2.data.haarcascades + haar_file

# Initialize the recognizer and load the trained model
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read(trainer_path)

# Load the Haar Cascade classifier
face_cascade = cv2.CascadeClassifier(haar_path)

# Define the names (must match the order of folders in datasets)
names = {0: 'ratan tata', 1: 'smriti_mandhana'}

print("\n [INFO] Image-based Face Recognition")
print(" [INFO] Model loaded successfully!")

def recognize_faces_in_folder(folder_path):
    """Recognize faces in all images from a folder"""
    if not os.path.isdir(folder_path):
        print(f" [ERROR] Folder not found: {folder_path}")
        return
    
    print(f"\n [INFO] Processing images from: {folder_path}\n")
    
    results = []
    
    for filename in os.listdir(folder_path):
        if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue
        
        image_path = os.path.join(folder_path, filename)
        img = cv2.imread(image_path)
        
        if img is None:
            print(f" [WARNING] Could not read: {filename}")
            continue
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 4)
        
        if len(faces) == 0:
            print(f" [{filename}] No faces detected")
            continue
        
        for (x, y, w, h) in faces:
            face = gray[y:y + h, x:x + w]
            label, confidence = recognizer.predict(face)
            
            person_name = names.get(label, "Unknown")
            confidence_pct = round(100 - confidence, 2)
            
            result = f" [{filename}] → {person_name} ({confidence_pct}%)"
            print(result)
            results.append(result)
            
            # Draw and save annotated image
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(img, f"{person_name} ({confidence_pct}%)", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Save annotated image
        output_path = os.path.join(script_dir, f"result_{filename}")
        cv2.imwrite(output_path, img)
    
    print(f"\n [INFO] Results saved in current directory with 'result_' prefix")

# Main execution
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        folder_to_process = sys.argv[1]
    else:
        # Default to current directory
        folder_to_process = script_dir
        print(f" [INFO] Usage: python recognize_from_folder.py <folder_path>")
        print(f" [INFO] Processing current directory by default...\n")
    
    recognize_faces_in_folder(folder_to_process)
    print("\n [INFO] Done!")

