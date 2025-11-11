import cv2
import os

trained_face_data = cv2.CascadeClassifier(
    'HaarcascadeFiles/haarcascade_frontalface_default.xml')
trained_smile_data = cv2.CascadeClassifier('HaarcascadeFiles/Smile.xml')

def detect_smile(image_path):
    
    img = cv2.imread(image_path)
    if img is None:
        return
        
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = trained_face_data.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5,
        minSize=(20, 20)
    )

    if len(faces) == 0:
        cv2.imshow('Smile Detection Result', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = img[y:y + h, x:x + w]

        smiles = trained_smile_data.detectMultiScale(
            roi_gray,
            scaleFactor=1.3, 
            minNeighbors=30,
            minSize=(15, 15)
        )

        if len(smiles) > 0:
            for (sx, sy, sw, sh) in smiles:
                cv2.rectangle(roi_color, (sx, sy), (sx + sw, sy + sh), (0, 255, 0), 2)
            cv2.putText(img, 'Sorriso', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow('Smile Detection Result', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


image_files = ['image1.jpg', 'image2.jpg', 'image3.jpg', 'image4.jpg', 'image5.jpg']

for file in image_files:
    detect_smile(file)