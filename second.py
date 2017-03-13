from collections import deque
import numpy as np
import argparse     #The argparse module makes it easy to write user-friendly command-line interfaces
import imutils      #image processing
import cv2
import math

ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", help="path to the (optional) video file") #_StoreAction(option_strings=['-v', '--video'], dest='video', nargs=None, const=None, default=None, type=None, choices=None, help='path to the (optional) video file', metavar=None)
ap.add_argument("-b", "--buffer", type=int, default=32, help="max buffer size") #_StoreAction(option_strings=['-b', '--buffer'], dest='buffer', nargs=None, const=None, default=32, type=<type 'int'>, choices=None, help='max buffer size', metavar=None)
args = vars(ap.parse_args())

greenLower = (29, 86, 6)
greenUpper = (64, 255, 255)

points = deque(maxlen=args["buffer"])
counter = 0
(dX, dY) = (0, 0)
direction = ""
if not args.get("video", False):
    camera = cv2.VideoCapture(0)

else:
    camera = cv2.VideoCapture(args["video"])

while True:
    (grabbed, frame) = camera.read()

    if args.get("video") and not grabbed:
        break
    frame = imutils.resize(frame, width=1000)
    # hand guesture
    cv2.rectangle(frame,(750,300),(900,150), (0, 255, 0), 0) #defining work area and cropping the image
    crop_img = frame[150:300, 750:900]
    cv2.imshow("crop",crop_img)
    grey = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
    value = (35, 35)
    blurred = cv2.GaussianBlur(grey, value, 0) #blur
    _, thresh1 = cv2.threshold(blurred, 127, 255,
                               cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    cv2.imshow('Thresholded', thresh1)                         #finding the threshhold

    contours, hierarchy = cv2.findContours(thresh1.copy(), cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
    cnt = max(contours, key=lambda x: cv2.contourArea(x))
    x, y, w, h = cv2.boundingRect(cnt)
    cv2.rectangle(crop_img, (x, y), (x + w, y + h), (0, 0, 255), 0)
    hull = cv2.convexHull(cnt)
    drawing = np.zeros(crop_img.shape, np.uint8)
    cv2.drawContours(drawing, [cnt], 0, (0, 255, 0), 0)
    cv2.drawContours(drawing, [hull], 0, (0, 0, 255), 0)
    hull = cv2.convexHull(cnt, returnPoints=False)
    defects = cv2.convexityDefects(cnt, hull)
    count_defects = 0
    cv2.drawContours(thresh1, contours, -1, (0, 255, 0), 3)
    for i in range(defects.shape[0]):
        s, e, f, d = defects[i, 0]
        start = tuple(cnt[s][0])
        end = tuple(cnt[e][0])
        far = tuple(cnt[f][0])
        a = math.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
        b = math.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
        c = math.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)
        angle = math.acos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c)) * 57    #angle abc
        if angle <= 90:
            count_defects += 1
            cv2.circle(crop_img, far, 1, [0, 0, 255], -1)
        dist = cv2.pointPolygonTest(cnt,far,True)
        cv2.line(crop_img, start, end, [0, 255, 0], 2)
        cv2.circle(crop_img,far,5,[0,0,255],-1)
    if count_defects == 1:
        print("two fingers")
    elif count_defects == 2:
        print("three fingers")
    elif count_defects == 3:
        print("four fingers")
    elif count_defects == 4:
        print("five fingers")
    else:
        print("one or zero fingers")
    cv2.imshow('Gesture', frame)
    all_img = np.hstack((drawing, crop_img))
    cv2.imshow('Contours', all_img)
    #motion tracking
    blurred = cv2.GaussianBlur(frame, (11, 11), 0)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(hsv, greenLower, greenUpper)     #hsv mode only identifies green color
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)
    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)[-2]
    center = None
    if len(cnts) > 0:
        c = max(cnts, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        M = cv2.moments(c)
        center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
        if radius > 10:
            cv2.putText(frame, str(radius), (200, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 3)
            cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), 2)
            cv2.circle(frame, center, 5, (0, 0, 255), -1)
            points.appendleft(center)
            print(points)
    for i in np.arange(1, len(points)):
        if points[i - 1] is None or points[i] is None:
            continue
        if counter >= 10 and i == 1 and len(points) == args["buffer"]:
            dX = points[-10][0] - points[i][0]
            dY = points[-10][1] - points[i][1]
            (dirX, dirY) = ("", "")

            if np.abs(dX) > 20:
                dirX = "RIGHT" if np.sign(dX) == 1 else "LEFT"

            if np.abs(dY) > 20:
                dirY = "UP" if np.sign(dY) == 1 else "DOWN"

            if dirX != "" and dirY != "":
                direction = "{}-{}".format(dirY, dirX)

            else:
                direction = dirX if dirX != "" else dirY

        thickness = int(np.sqrt(args["buffer"] / float(i + 1)) * 2.5)
        cv2.line(frame, points[i - 1], points[i], (0, 0, 255), thickness)
        if(dX!=0):
             angle2=0
             angle = int(math.atan(dY / dX) * 180 / math.pi)
             if angle!=angle2:
                 print angle
             angle2=angle

    cv2.putText(frame, direction, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 3)
    cv2.putText(frame, "dx: {}, dy: {}".format(dX, dY),
                (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    counter += 1

    if key == ord("q"):
        break
camera.release()
cv2.destroyAllWindows()
