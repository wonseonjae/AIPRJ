import cv2

face_cascade = cv2.CascadeClassifier("data/haarcascade_frontalface_alt2.xml")

gender_net = cv2.dnn.readNetFromCaffe(
    "model/deploy_gender.prototxt", "model/gender_net.caffemodel")

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)

gender_list = ["Male", "Female"]

image = cv2.imread("image/my_face.jpg", cv2.IMREAD_COLOR)

image = cv2.resize(image, (500, 500))

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

gray = cv2.equalizeHist(gray)

faces = face_cascade.detectMultiScale(gray, 1.3, 5, 0, (100, 100))

for face in faces:
    x, y, w, h = face

    face_image = image[y:y + h, x:x + w]

    blob = cv2.dnn.blobFromImage(face_image, 1, (227, 227), MODEL_MEAN_VALUES, swapRB=False)

    gender_net.setInput(blob)

    gender_preds = gender_net.forward()

    gender = gender_preds.argmax()

    cv2.rectangle(image, face, (255, 0, 0), 4)

    result = "Gender : " + gender_list[gender]

    cv2.putText(image, result, (x, y - 15), 0, 1, (255, 0, 0), 2)

cv2.imshow("myface", image)
cv2.waitKey(0)
