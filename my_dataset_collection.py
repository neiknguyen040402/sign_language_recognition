import cv2
import time
import os

label = "test"
path = r'C:\Users\Kien_PC\Documents\Workspace\ML\Sign_language' + '\\'
mode = 'train/'

if not os.path.exists(path + 'data/'):
    os.mkdir(path + 'data/')
if not os.path.exists(path + 'data/' + mode):
    os.mkdir(path + 'data/' + mode)

cap = cv2.VideoCapture(0)

i = 0
while(True):
    i += 1
    ret, frame = cap.read()
    if not ret:
        continue
    frame = cv2.flip(frame, 1)

    # Hiển thị
    cv2.rectangle(frame, (319, 9), (621, 309), (0, 255, 0), 1)
    roi = frame[10:300, 320:620]

    cv2.imshow("Frame", frame)

    # Lưu dữ liệu
    if i >= 60 and i <= 660:
        print("Image : ", i-60)
        # Tạo thư mục nếu chưa có
        if not os.path.exists(path + 'data/' + mode + str(label)):
            os.mkdir(path + 'data/' + mode + str(label))

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gaussblur = cv2.GaussianBlur(gray, (5, 5), 2)
        smallthres = cv2.adaptiveThreshold(gaussblur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 9,
                                           2.8)
        ret, final_image = cv2.threshold(smallthres, 70, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        final_image = cv2.resize(final_image, (300, 300))
        cv2.imshow("post-processing-image", final_image)

        cv2.imwrite(path + 'data/' + mode + str(label) + "/" + str(i - 60) + ".png", final_image)
        time.sleep(0.5)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()