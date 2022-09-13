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
            # face좌표 int로 형변환
            face_center = (int(x + w // 2)), int(y + h // 2)

            eye_centers = [[x + ex + ew // 2, y + ey + eh // 2] for ex, ey, ew, eh in eyes]

            correction_image, correction_center = doCorrectionImage(image, face_center, eye_centers)

            rois = doDetectObject(faces[0], face_center)

            cv2.rectangle(correction_image, rois[0], (255, 0, 255), 2)
            cv2.rectangle(correction_image, rois[1], (255, 0, 255), 2)

            cv2.rectangle(correction_image, rois[2], (255, 0, 255), 2)

            cv2.circle(correction_image, tuple(correction_center[0]), 5, (0, 255, 0), 2)
            cv2.circle(correction_image, tuple(correction_center[1]), 5, (0, 255, 0), 2)

            cv2.circle(correction_image, face_center, 3, (0, 0, 255), 2)

            cv2.imshow("MyFace_Edit", correction_image)

    else:
        print("눈 미검출")
else:
    print("얼굴 미검출")

cv2.waitKey(0)
