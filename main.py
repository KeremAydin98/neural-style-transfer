import cv2
from models import NeuralStyleTransfer


style_img = cv2.imread("Data/van-gogh.jpg")
content_img = cv2.imread("Data/me.jpg")

nst = NeuralStyleTransfer()

gen_img = nst.transfer(style_img, content_img)

cv2.imshow(gen_img)

cv2.waitKeys(0)
cv2.destroyAllWindows()