import cv2

image = cv2.imread("image/3NBHEFMDIMIOSLAPDCZDZ33ZWI.jpg", cv2.IMREAD_COLOR)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

gray = cv2.equalizeHist(gray)

if image is None: raise Exception("이미지 일기 실패")

face_cascade = cv2.CascadeClassifier("data/haarcascade_frontalface_alt2.xml")

faces = face_cascade.detectMultiScale(gray, 1.1, 5, 0,(100,100))

facesCnt = len(faces)

print(len(faces))

if facesCnt > 0:

    for face in faces:

        x,y,w,h = face

        face_image = image[y:y + h, x:x + w]

        mosaic_rate = 30

        face_image = cv2.resize(face_image, (w // mosaic_rate, h // mosaic_rate))

        face_image = cv2.resize(face_image, (w,h), interpolation=cv2.INTER_AREA)

        image[y:y + h, x:x + w] = face_image
    cv2.imwrite("result/my_image_mosaic.jpg", image)

    cv2.imshow("mosaic_image", cv2.imread("result/my_image_mosaic.jpg", cv2.IMREAD_COLOR))
else: print("얼굴 미검출")

cv2.waitKey(0)