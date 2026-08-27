import cv2
import numpy as np

def nothing(x): pass

img  = cv2.imread(r"C:\Users\barag\PycharmProjects\AI\capstone\eggplant_dataset\diseases\Fruit_rot\s1_test_IMG_20221109_094439_jpg.rf.c4ca4351391583bba162e0be500503e5.jpg")
hsv  = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

cv2.namedWindow("Tuner")
cv2.createTrackbar("H Low",  "Tuner",  5,  179, nothing)
cv2.createTrackbar("H High", "Tuner", 30,  179, nothing)
cv2.createTrackbar("S Low",  "Tuner", 40,  255, nothing)
cv2.createTrackbar("S High", "Tuner", 255, 255, nothing)
cv2.createTrackbar("V Low",  "Tuner", 20,  255, nothing)
cv2.createTrackbar("V High", "Tuner", 180, 255, nothing)

while True:
    lo = np.array([cv2.getTrackbarPos("H Low",  "Tuner"),
                   cv2.getTrackbarPos("S Low",  "Tuner"),
                   cv2.getTrackbarPos("V Low",  "Tuner")])
    hi = np.array([cv2.getTrackbarPos("H High", "Tuner"),
                   cv2.getTrackbarPos("S High", "Tuner"),
                   cv2.getTrackbarPos("V High", "Tuner")])

    mask   = cv2.inRange(hsv, lo, hi)
    result = cv2.bitwise_and(img, img, mask=mask)
    cv2.imshow("Tuner", result)
    cv2.imshow("Mask",  mask)

    if cv2.waitKey(1) == 27:  # ESC to quit
        print(f"Final values → Lower: {lo}  Upper: {hi}")
        break

cv2.destroyAllWindows()