import numpy as np
import cv2
import tensorflow as tf
from keras.models import load_model
from string import ascii_uppercase


gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=2048)])
    except RuntimeError as e:
        print(e)

# Load model
model = load_model('models/myModel.h5')

cam = cv2.VideoCapture(0)

alpha_dict = {}
j = 0
alpha_dict[j] = ''
j = 1
for i in ascii_uppercase:
   alpha_dict[j] = i
   j = j + 1


while True:
    _, frame = cam.read()

    frame = cv2.flip(frame, 1)

    cv2.rectangle(frame, (319, 9), (621, 309), (0, 255, 0), 1)
    # phần quan tâm roi (region of interest)
    roi = frame[10:300, 320:620]

    # xử lý ảnh bên trong roi
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gaussblur = cv2.GaussianBlur(gray, (5, 5), 2)
    smallthres = cv2.adaptiveThreshold(gaussblur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 9, 2.8)
    ret, final_image = cv2.threshold(smallthres, 70, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    cv2.imshow("post-processing-image", final_image)
    final_image = cv2.resize(final_image, (128, 128))
    final_image = np.reshape(final_image, (1, final_image.shape[0], final_image.shape[1], 1))
    # đưa ảnh vào predict
    pred = model.predict(final_image)

    if np.max(pred) >= 0.8:
        cv2.putText(frame, alpha_dict[np.argmax(pred)], (10, 50), cv2.FONT_HERSHEY_COMPLEX, 2, (40, 40, 239), 1)
    cv2.imshow("Cam", frame)

    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break

cam.release()
cv2.destroyAllWindows()