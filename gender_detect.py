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

            base_mask = np.full(correction_image.shape[:2], 255, np.uint8)

            face_mask = draw_ellipse(base_mask, rois[3], 0, -1)

            lip_mask = draw_ellipse(np.copy(base_mask), rois[2], 255)

            masks = [face_mask, face_mask, lip_mask, ~lip_mask]

            masks = [mask[y:y + h, x:x + w] for mask, (x, y, w, h) in zip(masks, rois)]

            subs = [correction_image[y:y + h, x:x + w] for x, y, w, h in rois]

            hists = [cv2.calcHist([sub], [0, 1, 2], mask, (120, 120, 120), (0, 256, 0, 256, 0, 256)) for sub, mask in zip(subs, masks)]

            hists = [h / np.sum(h) for h in hists]

            sim1 = cv2.compareHist(hists[3], hists[2], cv2.HISTCMP_CORREL)

            sim2 = cv2.compareHist(hists[0], hists[1], cv2.HISTCMP_CORREL)

            criteria = 0.2 if sim1 > 0.2 else 0.1

            value = sim2 > criteria

            text = "women" if value else "man"

            cv2.putText(image, text, (0, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

            cv2.imshow("MyFace", image)
    else:
        print("눈 미검출")
else:
    print("얼굴 미검출")

cv2.waitKey(0)
