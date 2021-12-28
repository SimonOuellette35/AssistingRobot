import cv2
import torch

MODEL_TYPE = 'person' # for the hands model, replace this with 'hands'

cam = cv2.VideoCapture(0)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 1024)

if MODEL_TYPE == 'person':
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
elif MODEL_TYPE == 'hands':
    model = torch.hub.load('ultralytics/yolov5', 'custom', path='models/hands_model.pt', force_reload=True)  # default
else:
    print("Error: unknown model type. Use either 'person' or 'hands'")
    exit(-1)
    
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