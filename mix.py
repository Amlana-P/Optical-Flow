import cv2
import numpy as np
cap = cv2.VideoCapture(0)
# Create old frame
_, frame = cap.read()
old_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
# Lucas kanade params
lk_params = dict(winSize=(15, 15),
                 maxLevel=4,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# color
color = (0, 255, 0)


# Mouse function
def select_point(event, x, y, flags, params):
    global point, point_selected, old_points
    if event == cv2.EVENT_LBUTTONDOWN:
        point = (x, y)
        point_selected = True
        old_points = np.array([[x, y]], dtype=np.float32)


cv2.namedWindow("Frame")
cv2.setMouseCallback("Frame", select_point)
point_selected = False
point = ()
old_points = np.array([[]])

# Create a mask image for drawing purposes
mask = np.zeros_like(old_gray)
new_points_list = []
old_points_list = []
while True:
    _, frame = cap.read()
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if point_selected is True:
        cv2.circle(frame, point, 5, (0, 0, 255), 2)
        new_points, status, error = cv2.calcOpticalFlowPyrLK(old_gray, gray_frame, old_points, None, **lk_params)
        new_points_list.append(new_points)
        old_points_list.append(old_points)
        for i, (new, old) in enumerate(zip(new_points_list, old_points_list)):
            a, b = new.ravel()
            c, d = old.ravel()
            print(a, b, c, d)
            frame1 = cv2.line(frame, (a, b), (c, d), (255, 0, 0), 2)
        frame2 = cv2.circle(frame, (a, b), 5, color, -1)
        img = cv2.add(frame1, frame2)
        cv2.imshow('Frame', img)
        old_gray = gray_frame.copy()
        old_points = new_points.reshape(-1, 1, 2)

    key = cv2.waitKey(1) & 0xff
    if key == 27:
        break
cap.release()
cv2.destroyAllWindows()
