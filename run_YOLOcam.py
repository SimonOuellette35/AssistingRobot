import cv2
import torch

cam = cv2.VideoCapture(0)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 1024)

model = torch.hub.load('ultralytics/yolov5', 'yolov5s6')

# The lines below are to load the hands detection model
# checkpoint_ = torch.load('models/hands_model.pt')['model']
# model.load_state_dict(checkpoint_.state_dict())

model.conf = 0.25

first = True
while True:

    ret, img = cam.read()

    results = model(img)

    imgs = results.render()

    for img in imgs:
        cv2.imshow("", img)

    if (cv2.waitKey(1) & 0xFF == ord('q')) or (cv2.waitKey(1)==27):
        break

cam.release()
cv2.destroyAllWindows()