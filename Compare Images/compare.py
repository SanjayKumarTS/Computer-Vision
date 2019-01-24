# import the necessary packages
from skimage.measure import compare_ssim
import argparse
import imutils
import cv2
from skimage import feature
import matplotlib.pyplot as plt


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-f", "--first", required=True,
                help="first input image")
ap.add_argument("-s", "--second", required=True,
                help="second")
args = vars(ap.parse_args())

# load the two input images
imageA = cv2.imread(args["first"])
imageB = cv2.imread(args["second"])

# convert the images to grayscale
# grayA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
# grayB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)

grayA = cv2.imread(args["first"], 0)
grayB = cv2.imread(args["second"], 0)

# compute the Structural Similarity Index (SSIM) between the two
# images, ensuring that the difference image is returned

(score, diff) = compare_ssim(grayA, grayB, full=True)
diff = (diff * 255).astype("uint8")
print("SSIM: {}".format(score))
# threshold the difference image, followed by finding contours to
# obtain the regions of the two input images that differ
thresh = cv2.threshold(diff, 128, 255,
                       cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                        cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
# loop over the contours
for c in cnts:
    # compute the bounding box of the contour and then draw the
    # bounding box on both input images to represent where the two
    # images differ
    (x, y, w, h) = cv2.boundingRect(c)
    cv2.rectangle(imageA, (x, y), (x + w, y + h), (0, 0, 255), 2)
    cv2.rectangle(imageB, (x, y), (x + w, y + h), (0, 0, 255), 2)

# edges1 = feature.canny(grayA, sigma=3)
# edges2 = feature.canny(grayB, sigma=3)
# fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(8, 3),
#                                     sharex=True, sharey=True)
# ax2.imshow(edges1, cmap=plt.cm.gray)
# ax2.axis('off')
# ax2.set_title('Canny filter1', fontsize=20)
# plt.show()

# ax2.imshow(edges2, cmap=plt.cm.gray)
# ax2.axis('off')
# ax2.set_title('Canny filter2', fontsize=20)
# plt.show()
# show the output images

cv2.imshow("Original", imageA)
cv2.imshow("Modified", imageB)
cv2.imshow("Diff", diff)
cv2.imshow("Thresh", thresh)
cv2.waitKey(0)