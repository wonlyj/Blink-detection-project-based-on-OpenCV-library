import cv2
import random
import numpy as np

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

def detect_blink(frame, roi_x, roi_y, roi_w, roi_h, dog, prev_eye_size_ratio):
    counter = 0
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray)

    if len(faces) >= 1:
        for (x, y, w, h) in faces:
            dog_resized = cv2.resize(dog, (w, h))
            frame[y:y + h, x:x + w] = dog_resized

        if roi_x > 0 and roi_y > 0 and roi_x + roi_w < frame.shape[1] and roi_y + roi_h < frame.shape[0]:
            roi_gray = gray[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w]
            eyes = eye_cascade.detectMultiScale(roi_gray)

            if len(eyes) >= 2:
                # # 获取两只眼睛的高度
                # eye1_h = eyes[0][3]
                # eye2_h = eyes[1][3]
                eye1 = eyes[0]
                eye2 = eyes[1]
                _, _, _, eye1_h = eye1
                _, _, _, eye2_h = eye2
                eye_size_ratio = eye2_h/eye1_h
                print(eye_size_ratio)
                if abs(eye_size_ratio - prev_eye_size_ratio) > 0.5:
                    counter += 1
                if counter >= 1:
                    new_dog = random.choice(dogs)
                    while np.array_equal(new_dog, dog):
                        new_dog = random.choice(dogs)
                    dog = new_dog
                prev_eye_size_ratio = eye_size_ratio

    return frame, dog, prev_eye_size_ratio


def reduce_noise(frame):
    # 使用高斯滤波器平滑图像
    smoothed = cv2.GaussianBlur(frame, (5, 5), 0)
    return smoothed


cap = cv2.VideoCapture(0)
cv2.namedWindow('Blink Detection', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Blink Detection', 800, 600)

Dog1 = cv2.imread('./img/1.jpg')
Dog2 = cv2.imread('./img/2.jpg')
Dog3 = cv2.imread('./img/3.jpg')
Dog4 = cv2.imread('./img/4.jpg')
dogs = [Dog1, Dog2, Dog3, Dog4]
# # 加载四张狗的图片
# dogs = [cv2.imread(f'./img/{i}.jpg') for i in range(1, 5)]

roi_x = 150
roi_y = 100
roi_w = 300
roi_h = 300

dog = random.choice(dogs)
prev_eye_size_ratio = 1

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    frame = reduce_noise(frame)
    frame, dog, prev_eye_size_ratio = detect_blink(frame, roi_x, roi_y, roi_w, roi_h, dog, prev_eye_size_ratio)
    cv2.rectangle(frame, (roi_x, roi_y), (roi_x + roi_w, roi_y + roi_h), (0, 255, 0), 2)
    cv2.imshow('Blink Detection', frame)
    if cv2.waitKey(1000 // 30) == 27:
        break

cap.release()
cv2.destroyAllWindows()
