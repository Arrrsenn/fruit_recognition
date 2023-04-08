import cv2
import os

path = "C:\\Users\\alevo\\PycharmProjects\\lab1\\images"

img_watermelon = cv2.imread("img2.jpg")
img_watermelon2 = cv2.imread("img.jpg")
img_apple = cv2.imread("img_1.jpg")

t_lower = 150
t_upper = 500

edge_watermelon = cv2.Canny(img_watermelon, t_lower, t_upper)
edge_watermelon1 = cv2.Canny(img_watermelon2, t_lower, t_upper)
edge_apple = cv2.Canny(img_apple, t_lower, t_upper)

os.chdir(path)
filename_apple = 'apple.jpg'
filename_watermelon = 'watermelon.jpg'
filename_watermelon2 = 'watermelon_2.jpg'

cv2.imwrite(filename_apple, edge_apple)
cv2.imwrite(filename_watermelon, edge_watermelon)
# cv2.imwrite(filename_watermelon2, edge_watermelon1)

cv2.waitKey(0)
cv2.destroyAllWindows()
