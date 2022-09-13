import numpy as np, cv2


def preprocessing():
    image = cv2.imread("image/my_face.jpg", cv2.IMREAD_COLOR)

    if image is None: return None, None

    image = cv2.resize(image, (700, 700))

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    gray = cv2.equalizeHist(gray)

    face_cascade = cv2.CascadeClassifier("data/haarcascade_frontalface_alt2.xml")

    eye_cascade = cv2.CascadeClassifier("data/haarcascade_eye_tree_eyeglasses.xml")

    return image, gray, face_cascade, eye_cascade


def doCorrectionImage(image, face_center, eye_centers):
    pt0, pt1 = eye_centers

    if pt0[0] > pt1[0]: pt0, pt1 = pt1, pt0

    dx, dy = np.subtract(pt1, pt0).astype(float)

    angle = cv2.fastAtan2(dy, dx)

    rot = cv2.getRotationMatrix2D(face_center, angle, 1)

    size = image.shape[1::-1]

    correction_image = cv2.warpAffine(image, rot, size, cv2.INTER_CUBIC)

    eye_centers = np.expand_dims(eye_centers, axis=0)
    correction_centers = cv2.transform(eye_centers, rot)
    correction_centers = np.squeeze(correction_centers, axis=0)

    return correction_image, correction_centers


def doDetectObject(face, center):
    w, h = face[2:4]
    center = np.array(center)

    face_avg_rate = np.multiply((w, h), (0.45, 0.65))

    lib_avg_rate = np.multiply((w, h), (0.18, 0.1))

    pt1 = center - face_avg_rate

    pt2 = center + face_avg_rate

    face_all = define_roi(pt1, pt2 - pt1)

    # 평균적으로 얼굴 총 범위의 35%정도가 윗머리
    size = np.multiply(face_all[2:4], (1, 0.35))

    face_up = define_roi(pt1, size)

    face_down = define_roi(pt2 - size, size)

    lip_center = center + (0, h * 0.3)

    lip1 = lip_center - lib_avg_rate

    lip2 = lip_center + lib_avg_rate

    lip = define_roi(lip1, lip2 - lip1)

    return [face_up, face_down, lip, face_all]


def draw_ellipse(image, roi, color, thickness=cv2.FILLED):
    x, y, w, h = roi

    center = (x + w // 2, y + h // 2)

    size = (int(w * 0.45), int(h * 0.45))

    cv2.ellipse(image, center, size, 0, 0, 360, color, thickness)

    return image


def define_roi(pt, size):
    return np.ravel([pt, size]).astype(int)
