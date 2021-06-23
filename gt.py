import cv2
import numpy as np
import dlib
from math import hypot
from heatmap import heatmap

cap = cv2.VideoCapture(0)

detector = dlib.get_frontal_face_detector()

predictor = dlib.shape_predictor("data/shape_predictor_68_face_landmarks.dat")

font = cv2.FONT_HERSHEY_PLAIN

def midpoint(p1, p2):
    return int((p1.x + p2.x) / 2), int((p1.y + p2.y) /2)

def get_eye(eye_points, facial_landmarks):
    left_point = (facial_landmarks.part(eye_points[0]).x, facial_landmarks.part(eye_points[0]).y)
    right_point = (facial_landmarks.part(eye_points[3]).x, facial_landmarks.part(eye_points[3]).y)

    center_top = midpoint(facial_landmarks.part(eye_points[1]), facial_landmarks.part(eye_points[2]))
    center_bottom = midpoint(facial_landmarks.part(eye_points[5]), facial_landmarks.part(eye_points[4]))

    hor_line_length = hypot((left_point[0] - right_point[0]), (left_point[1] - right_point[1]))
    ver_line_length = hypot((center_top[0] - center_bottom[0]), (center_top[1] - center_bottom[1]))

    eye_x = int((left_point[0] + right_point[0])/2)
    eye_y = int((center_top[1] + center_bottom[1])/2)

    cv2.circle(frame, (eye_x, eye_y), 5, (0, 0, 255), 2)

    ratio = hor_line_length / ver_line_length

    return ratio

def get_gaze_ratio(eye_points,facial_landmarks):
    # Gaze Detection
    eye_region = np.array([
        (facial_landmarks.part(eye_points[0]).x, facial_landmarks.part(eye_points[0]).y),
        (facial_landmarks.part(eye_points[1]).x, facial_landmarks.part(eye_points[1]).y),
        (facial_landmarks.part(eye_points[2]).x, facial_landmarks.part(eye_points[2]).y),
        (facial_landmarks.part(eye_points[3]).x, facial_landmarks.part(eye_points[3]).y),
        (facial_landmarks.part(eye_points[4]).x, facial_landmarks.part(eye_points[4]).y),
        (facial_landmarks.part(eye_points[5]).x, facial_landmarks.part(eye_points[5]).y)], np.int32)

    h,w,_ = frame.shape
    mask = np.zeros((h,w), np.uint8)

    cv2.polylines(mask, [eye_region], True, 255, 2)
    cv2.fillPoly(mask, [eye_region], 255)
    eye = cv2.bitwise_and(gray, gray, mask=mask)

    min_x = np.min(eye_region[:,0])
    max_x = np.max(eye_region[:,0])
    min_y = np.min(eye_region[:,1])
    max_y = np.max(eye_region[:,1])

    gray_eye = eye[min_y: max_y, min_x: max_x]
    _, threshold_eye = cv2.threshold(gray_eye, 40, 255, cv2.THRESH_BINARY)
    h, w = threshold_eye.shape

    left_side_threshold = threshold_eye[0:h+5, 0:int(w/2)]
    left_side_white = cv2.countNonZero(left_side_threshold)
    
    right_side_threshold = threshold_eye[0:h+5, int(w/2): w]
    right_side_white = cv2.countNonZero(right_side_threshold)

    # Avoid Division by Zero
    if left_side_white <= 0:
        left_side_white = 0.1
    if right_side_white <= 0:
        right_side_white = 0.1
    
    gaze_ratio = left_side_white/right_side_white

    return gaze_ratio


counter = 0
heatmap_data = {'top_left': 0,'mid_left': 0,'down_left': 0,
                'top_center': 0,'mid_center': 0,'down_center': 0,
                'top_right': 0,'mid_right': 0,'down_right': 0}

while True:
    _, frame = cap.read()

    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = detector(gray)
    for face in faces:
        x,y = face.left(), face.top()
        x1, y1 = face.right(), face.bottom()
        
        landmarks = predictor(gray, face)

        left_eye_ratio = get_eye([36,37,38,39,40,41], landmarks)
        right_eye_ratio = get_eye([42,43,44,45,46,47], landmarks)

        gaze_ratio_left_eye = get_gaze_ratio([36,37,38,39,40,41], landmarks)
        gaze_ratio_right_eye = get_gaze_ratio([42,43,44,45,46,47], landmarks)        
        gaze_ratio = (gaze_ratio_left_eye + gaze_ratio_right_eye) / 2

        if gaze_ratio < 0.3:
            #cv2.putText(frame, "TOP LEFT", (50,100), font, 2, (0,0,255), 3)
            cv2.circle(frame,(30,30), 3, (255,0,0), 10)
            heatmap_data['top_left'] += 1
        elif 0.3 < gaze_ratio < 0.8:
            #cv2.putText(frame, "MID LEFT", (50,100), font, 2, (0,0,255), 3)
            cv2.circle(frame,(30,250), 3, (255,0,0), 10)
            heatmap_data['mid_left'] += 1
        elif 0.8 < gaze_ratio < 1.36:
            #cv2.putText(frame, "DOWN LEFT", (50,100), font, 2, (0,0,255), 3)
            cv2.circle(frame,(30,450), 3, (255,0,0), 10)
            heatmap_data['down_left'] += 1
        elif 1.36 < gaze_ratio < 1.92:
            #cv2.putText(frame, "TOP CENTER", (50,100), font, 2, (0,0,255), 3)
            cv2.circle(frame,(300,30), 3, (255,0,0), 10)
            heatmap_data['top_center'] += 1
        elif 1.92 < gaze_ratio < 6:
            #cv2.putText(frame, "MID CENTER", (50,100), font, 2, (0,0,255), 3)
            cv2.circle(frame,(300,250), 3, (255,0,0), 10)
            heatmap_data['mid_center'] += 1
        elif 6 < gaze_ratio < 11:
            #cv2.putText(frame, "DOWN CENTER", (50,100), font, 2, (0,0,255), 3)
            cv2.circle(frame,(300,250), 3, (255,0,0), 10)
            heatmap_data['down_center'] += 1
        elif 11 < gaze_ratio < 19:
            #cv2.putText(frame, "TOP RIGHT", (50,100), font, 2, (0,0,255), 3)
            cv2.circle(frame,(550,30), 3, (255,0,0), 10)
            heatmap_data['top_right'] += 1
        elif 19 < gaze_ratio < 56:
            #cv2.putText(frame, "MID RIGHT", (50,100), font, 2, (0,0,255), 3)
            cv2.circle(frame,(550,250), 3, (255,0,0), 10)
            heatmap_data['mid_right'] += 1
        else:
            #cv2.putText(frame, "DOWN RIGHT", (50,100), font, 2, (0,0,255), 3)
            cv2.circle(frame,(550,450), 3, (255,0,0), 10)
            heatmap_data['down_right'] += 1

        #cv2.putText(frame, str(gaze_ratio), (50,150), font, 2, (0,0,255), 3)

        print(gaze_ratio)

    cv2.imshow("Frame", frame)

    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()


heatmap(heatmap_data)


