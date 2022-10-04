import cv2

face_cascade = cv2.CascadeClassifier("data/haarcascade_frontalface_alt2.xml")

vcp = cv2.VideoCapture(0, cv2.CAP_DSHOW)

model = cv2.face.LBPHFaceRecognizer_create()

model.read("model/face-trainner2.yml")

while True:
    ret, my_image = vcp.read()

    if ret is True:

        gray = cv2.cvtColor(my_image, cv2.COLOR_BGR2GRAY)

        gray = cv2.equalizeHist(gray)

        faces = face_cascade.detectMultiScale(gray, 1.1, 5, 0, (20, 20))

        faceCnt = len(faces)

        if faceCnt == 1:

            x, y, w, h = faces[0]

            face_image = gray[y:y + h, x:x + w]

            id_, res = model.predict(face_image)

            result = "result : " + str(res) + "%"

            cv2.putText(my_image, result, (x, y - 15), 0, 1, (255, 0, 0), 2)

            cv2.rectangle(my_image, faces[0], (255, 0, 0), 4)
        cv2.imshow("predict_my_face", my_image)

    if cv2.waitKey(1) > 0:
        break

cv2.destroyAllWindows()