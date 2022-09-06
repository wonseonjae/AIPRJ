import cv2

image = cv2.imread("image/nam.jpg", cv2.IMREAD_COLOR)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

gray = cv2.equalizeHist(gray)

crown_image = cv2.imread("image/tiara.png", cv2.IMREAD_COLOR)

if image is None: raise Exception("이미지 읽기 실패")

face_casacade = cv2.CascadeClassifier("data/haarcascade_frontalface_alt2.xml")

faces = face_casacade.detectMultiScale(gray, 1.1, 5, 0, (100, 100))

faceCnt = len(faces)

if faceCnt > 0:
    for face in faces:
        x, y, w, h = face

        face_image = cv2.resize(crown_image, (w,120), interpolation=cv2.INTER_AREA)
        print(y+h)
        print(y+h+60)
        image[h-y:h-y+120, x:x + w] = face_image

    cv2.imwrite("result/crown.jpg", image)

    cv2.imshow("crown.jpg", cv2.imread("result/crown.jpg", cv2.IMREAD_COLOR))
else:
    print("얼굴 미검출")

cv2.waitKey(0)
