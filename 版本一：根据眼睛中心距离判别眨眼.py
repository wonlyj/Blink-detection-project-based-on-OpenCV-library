import cv2
from scipy.spatial import distance
import random
import numpy as np

def euclidean_dist(pt1, pt2):
    return distance.euclidean(pt1, pt2)

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

def detect_blink(frame, roi_x, roi_y, roi_w, roi_h, prev_eye_dist, dog, blink_counter, blink_threshold):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray)

    if len(faces) >= 1:
        for (x, y, w, h) in faces:
            dog_resized = cv2.resize(dog, (w, h))
            frame[y:y + h, x:x + w] = dog_resized

        if roi_x > 0 and roi_y > 0 and roi_x + roi_w < frame.shape[1] and roi_y + roi_h < frame.shape[0]:
            roi_gray = gray[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w]
            eyes = eye_cascade.detectMultiScale(roi_gray)

            if len(eyes) == 2:
                eye1 = eyes[0]
                eye2 = eyes[1]

                (ex1, ey1, ew1, eh1) = eye1
                (ex2, ey2, ew2, eh2) = eye2

                eye1_center = (int(ex1 + 0.5 * ew1), int(ey1 + 0.5 * eh1))
                eye2_center = (int(ex2 + 0.5 * ew2), int(ey2 + 0.5 * eh2))

                eye_dist = euclidean_dist(eye1_center, eye2_center)
                print(eye_dist)

                if prev_eye_dist > 0:
                    delta_dist = abs(eye_dist - prev_eye_dist)
                    if delta_dist > blink_threshold:
                        blink_counter += 1

                if blink_counter >= 1 :
                    new_dog = random.choice(dogs)
                    while np.array_equal(new_dog, dog):
                        new_dog = random.choice(dogs)
                    dog = new_dog
                    blink_counter = 0
                elif blink_counter > 1:
                    blink_counter = 0

                prev_eye_dist = eye_dist

    return frame, prev_eye_dist, dog, blink_counter


def reduce_noise(frame):
    # 使用高斯滤波器平滑图像
    smoothed = cv2.GaussianBlur(frame, (5, 5), 0)
    return smoothed

cap = cv2.VideoCapture(0)
#设置窗口大小
cv2.namedWindow('Blink Detection', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Blink Detection', 800, 600)

Dog1 = cv2.imread('./img/1.jpg')
Dog2 = cv2.imread('./img/2.jpg')
Dog3 = cv2.imread('./img/3.jpg')
Dog4 = cv2.imread('./img/4.jpg')
dogs = [Dog1, Dog2, Dog3,Dog4]
# dogs = [cv2.imread(f'./img/{i}.jpg') for i in range(1, 5)]
#设置检测区域
roi_x = 150
roi_y = 100
roi_w = 300
roi_h = 300

dog = random.choice(dogs)
prev_eye_dist = -1  # 初始化,一开始没有眼睛数据
blink_counter = 0
blink_threshold = 15  # 设定变化波动阈值，依据不同情况而设定

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    frame = reduce_noise(frame)
    frame, prev_eye_dist, dog, blink_counter = detect_blink(frame, roi_x, roi_y, roi_w, roi_h, prev_eye_dist, dog, blink_counter, blink_threshold)
    cv2.rectangle(frame, (roi_x, roi_y), (roi_x + roi_w, roi_y + roi_h), (0, 255, 0), 2)
    cv2.imshow('Blink Detection', frame)
    if cv2.waitKey(1000//30) == 27:
        break

cap.release()
cv2.destroyAllWindows()
