import cv2
from utill.CommUtils import *

image, gray, face_cascade, eye_cascade = preprocessing()

if image is None: raise Exception("이미지 파일 읽기 에러")

faces = face_cascade.detectMultiScale(gray, 1.1, 2, 0, (100, 100))

if faces.any():

    x, y, w, h = faces[0]

    face_image = image[y:y + h, x:x + w]

    eyes = eye_cascade.detectMultiScale(face_image, 1.15, 7, 0, (25, 20))

    if len(eyes) == 2:
        for ex, ey, ew, eh in eyes:

            center = (x + ex + ew // 2, y + ey + eh // 2)

            cv2.circle(image, center, 10, (0, 255, 0), 2)

    else:
        print("눈 미검출")

    cv2.rectangle(image, faces[0], (255, 0, 0), 4)

    cv2.imshow("MyFace", image)


else: print("얼굴 미검출")

cv2.waitKey(0)
