import cv2

img = cv2.imread("./object_data/IMG_1025.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
th, img = cv2.threshold(img, 0, 255, cv2.THRESH_OTSU)

cnt, labels, stats, centroids = cv2.connectedComponentsWithStats(img, 8)
dst = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

object_cnt = 0
for i in range(1, cnt):
    (x, y, w, h, area) = stats[i]
    if  500000 > area > 20000:
        print(stats[i])
        object_cnt += 1
        cv2.rectangle(dst, (x, y), (x + w, y + h), (0, 255, 255), 30)

print(object_cnt)
cv2.imwrite('./result.jpg', dst)