import cv2
import sys
import numpy as np
import pytesseract
from PIL import Image

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
    i = img
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

    config = ('-l eng --oem 3 --psm 3')

    # Recognize text with tesseract for python
    result = pytesseract.image_to_string(img, config=config)

    # Print recognized text
    print(result)
    # To read the coordinates
    cv2.imshow('image', img)
    cv2.imwrite('output.png', img)
    boxes = pytesseract.image_to_boxes(Image.open('output.png'))

    h, w = img.shape

    for b in boxes.splitlines():
        b = b.split(' ')
        img = cv2.rectangle(
            i, (int(b[1]), h - int(b[2])), (int(b[3]), h - int(b[4])), (255, 0, 0), 2)
    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.imshow('image', i)
    cv2.waitKey(0)