import grpc
import YOLOsvc_pb2
import YOLOsvc_pb2_grpc
import cv2
import base64

MODEL_TYPE = 'person' # for the hands model, replace this with 'hands'
MAX_MESSAGE_LENGTH = 6000000
WIDTH=640
HEIGHT=480

cam = cv2.VideoCapture(0)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)

def getObjectDetection(stub, img):

    x = base64.b64encode(img)

    request = YOLOsvc_pb2.ImageB64(b64image=x, width=WIDTH, height=HEIGHT)

    return stub.ObjectDetectionV2(request)

with grpc.insecure_channel('localhost:50051', options=[
    ('grpc.max_send_message_length', MAX_MESSAGE_LENGTH),
    ('grpc.max_receive_message_length', MAX_MESSAGE_LENGTH),
]) as channel:
    stub = YOLOsvc_pb2_grpc.YOLOsvcStub(channel)

    first = True
    while True:

        ret, img = cam.read()

        results = getObjectDetection(stub, img)
        print("Detected classes: ", results.classes)

        # TOOD: how to render detection box and show detection image?
        #imgs = results.render()

        # for img in imgs:
        #     cv2.imshow("", img)

        if (cv2.waitKey(1) & 0xFF == ord('q')) or (cv2.waitKey(1)==27):
            break

cam.release()
cv2.destroyAllWindows()