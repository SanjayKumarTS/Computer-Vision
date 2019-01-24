import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# load the image, convert it to grayscale, and blur it slightly
size = float(input("Enter the Size of the object(mm): "))
image = cv2.imread("images/1.bmp")
cv2.imshow('Image', cv2.resize(image.copy(), (960, 540)))
cv2.waitKey(0)
imgplot = plt.imshow(image)
plt.show()

# x1 = int(input("Enter The coordinate of x1: "))
# y1 = int(input("Enter The coordinate of y1: "))
# x2 = int(input("Enter The coordinate of x2: "))
# y2 = int(input("Enter The coordinate of y2: "))


def measure(x1, y1, x2, y2):
    global image
    print(x1)
    print(y1)
    print(x2)
    print(y2)
    boundRect = cv2.rectangle(
        image.copy(), (x1, y1), (x2, y2), (255, 0, 0), 3)

    cv2.imshow("ROI Image", cv2.resize(boundRect, (960, 540)))
    cv2.waitKey(0)

    # cv2.imshow("Image", image[300:400, 300:625])
    img = image[y1:y2, x1:x2]

    # image = image[300:400, 300:625]
    orig = img.copy()
    # orig = image[300:400, 300:625]
    gray = cv2.cvtColor(orig.copy(), cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)

    # perform edge detection, then perform a dilation + erosion to
    # close gaps in between object edges
    edged = cv2.Canny(gray, 80, 120)
    cv2.imshow("edged", edged)
    cv2.waitKey(0)
    edged = cv2.threshold(gray, 128, 255,
                          cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    edged = cv2.dilate(edged, None, iterations=1)
    edged = cv2.erode(edged, None, iterations=1)

    # find contours in the edge map
    cnts, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
                               cv2.CHAIN_APPROX_SIMPLE)
    print('Area: ', cv2.contourArea(cnts[0]))

    for cnt in cnts:
        area = cv2.contourArea(cnt)
        if area > 0:
            (x, y, w, h) = cv2.boundingRect(cnt)
            cv2.rectangle(orig, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.imshow("Image", cv2.resize(orig.copy(), (960, 540)))
        cv2.waitKey(0)
        extLeft = tuple(cnt[cnt[:, :, 0].argmin()][0])
        extRight = tuple(cnt[cnt[:, :, 0].argmax()][0])
        extTop = tuple(cnt[cnt[:, :, 1].argmin()][0])
        extBot = tuple(cnt[cnt[:, :, 1].argmax()][0])
        print("extLeft", extLeft)
        print("extRight", extRight)
        print("extTop", extTop)
        print("extBot", extBot)
        cv2.circle(orig, extLeft, 8, (0, 255, 0), -1)
        cv2.circle(orig, extRight, 8, (255, 255, 0), -1)
        cv2.circle(orig, extTop, 8, (255, 0, 0), -1)
        cv2.circle(orig, extBot, 8, (0, 0, 255), -1)
        width = extRight[0] - extLeft[0]
        height = extBot[1] - extTop[1]
        print("\nWidth = ", width, "px")
        print("\nHeight = ", height, "px")
        cv2.imshow("Image", cv2.resize(orig.copy(), (960, 540)))
        cv2.waitKey(0)

    pixel_metric = 918/size
    width_mm = width/pixel_metric
    height_mm = height/pixel_metric

    cv2.line(orig, (extLeft[0], (extTop[1]+extBot[1])//2),
             (extRight[0], (extTop[1]+extBot[1])//2), (0, 255, 0), 2)
    cv2.line(orig, ((extLeft[0]+extRight[0])//2, extTop[1]),
             ((extLeft[0]+extRight[0])//2, extBot[1]), (0, 255, 100), 2)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(orig, "{:.2f}mm".format(width_mm), (width//2 + extLeft[0]+30, (extTop[1]+extBot[1])//2-10), font,
                0.5, (200, 255, 155), 2, cv2.LINE_AA)
    cv2.putText(orig, "{:.2f}mm".format(height_mm), (width//2 + extLeft[0]-80, height//2-30), font,
                0.5, (200, 255, 155), 2, cv2.LINE_AA)
    # cv2.putText(orig, '20 mm', (extLeft[0], width//2),
    #             cv2.FONT_HERSHEY_SIMPLEX, 4, (255, 255, 255))
    print("\nWidth: ", width_mm, "mm")
    print("\nHeight: ", height_mm, "mm")
    cv2.imshow("Image", cv2.resize(orig.copy(), (960, 540)))
    cv2.waitKey(0)


x1 = [1005, 1160, 1184]
x2 = [2902, 1400, 2700]
y1 = [867, 1269, 2500]
y2 = [1292, 2578, 2663]

# x1 = 1323
# x2 = 2600
# y1 = 2516
# y2 = 2632

measure(x1[2], y1[2], x2[2], y2[2])

# for a1, b1, a2, b2 in zip(x1, y1, x2, y2):
#     measure(a1, b1, a2, b2)
