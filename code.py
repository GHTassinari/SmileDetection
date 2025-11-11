import cv2
import os

trained_smile_data = cv2.CascadeClassifier('HaarcascadeFiles/Smile.xml')
trained_face_data = cv2.CascadeClassifier(
    'HaarcascadeFiles/haarcascade_frontalface_default.xml')

image_files = ['image1.jpg', 'image2.jpg', 'image3.jpg', 'image4.jpg', 'image5.jpg']

for file in image_files:
    img = cv2.imread(file)
    if img is None:
        continue

    grayscaled_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)

    for (x, y, width, height) in face_coordinates:
        
        roi_gray = grayscaled_img[y:y+height, x:x+width]

        smile_coordinates = trained_smile_data.detectMultiScale(
            roi_gray, scaleFactor=1.7, minNeighbors=20)

        if len(smile_coordinates) > 0:
            
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(img, 'Sorriso', (x, y - 10), font, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
            
            for (a, b, c, d) in smile_coordinates:
                cv2.rectangle(img, (x + a, y + b), (x + a + c, y + b + d), (0, 255, 0), 2)

    new_img = cv2.resize(img, (720, 480))
    cv2.imshow("Image Smile Detection", new_img)
    
    key = cv2.waitKey(0)

    if key == 81 or key == 113:
        break

cv2.destroyAllWindows()