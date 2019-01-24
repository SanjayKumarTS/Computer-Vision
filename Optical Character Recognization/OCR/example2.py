# import csv
# import cv2
# from pytesseract import pytesseract as pt

# pt.run_tesseract('3.png', 'output', lang=None, boxes=True, config="hocr")

# # To read the coordinates
# boxes = []
# with open('output.box', 'rb') as f:
#     reader = csv.reader(f, delimiter=' ')
#     for row in reader:
#         if(len(row) == 6):
#             boxes.append(row)

# # Draw the bounding box
# img = cv2.imread('3.png')
# h, w, _ = img.shape
# for b in boxes:
#     img = cv2.rectangle(
#         img, (int(b[1]), h-int(b[2])), (int(b[3]), h-int(b[4])), (255, 0, 0), 2)

# cv2.imshow('output', img)

import cv2
import sys
import numpy as np
import pytesseract
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

if __name__ == '__main__':

    if len(sys.argv) < 2:
        print('Usage: python ocr_simple.py image.jpg')
        sys.exit(1)

    # Read image path from command line
    imPath = sys.argv[1]

    # Uncomment the line below to provide path to tesseract manually
    # pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'

    # Define config parameters.
    # '-l eng'  for using the English language
    # '--oem 1' for using LSTM OCR Engine

    # Read image using opencv
    img = cv2.imread(imPath)

    # Rescale the image, if needed.
    img = cv2.resize(img, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)

    # Convert to gray
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply dilation and erosion to remove some noise
    kernel = np.ones((1, 1), np.uint8)
    img = cv2.dilate(img, kernel, iterations=1)
    img = cv2.erode(img, kernel, iterations=1)

    # Apply blur to smooth out the edges
    img = cv2.GaussianBlur(img, (5, 5), 0)

    # Apply threshold to get image with only b&w (binarization)
    img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    config = ('-l eng --oem 3 --psm 3 hocr')

    # Recognize text with tesseract for python
    result = pytesseract.image_to_string(img, config=config)

    # Print recognized text
    print(result)
    h, w = img.shape
    # To read the coordinates
    cv2.imshow('image', img)
    cv2.imwrite('output.png', img)
    boxes = pytesseract.image_to_boxes(Image.open('output.png'))
    print(boxes)
    # im = np.array(Image.open('output.png'), dtype=np.uint8)
    # fig, ax = plt.subplots(1)

    # # Display the image
    # ax.imshow(im)

    # # Create a Rectangle patch
    # rect = patches.Rectangle((416, h-148), 29, 35, linewidth=1,
    #                          edgecolor='r', facecolor='none')

    # # Add the patch to the Axes
    # ax.add_patch(rect)

    # plt.show()

    for b in boxes.splitlines():
        b = b.split(' ')
        img = cv2.rectangle(
            img, (int(b[1]), h - int(b[2])), (int(b[3]), h - int(b[4])), (0, 255, 0), 2)
    cv2.imshow('image', img)
    cv2.waitKey(0)
