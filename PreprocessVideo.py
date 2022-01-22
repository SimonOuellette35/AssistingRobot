import numpy as np
import cv2
import csv
import grpc
import YOLOsvc_pb2
import YOLOsvc_pb2_grpc
import base64

timestamp = '2022-01-22_23-40-47'
MAX_MESSAGE_LENGTH = 6000000
WIDTH = 640
HEIGHT = 480

# load frames one by one from video
cap = cv2.VideoCapture('data/%s.avi' % timestamp)

out = cv2.VideoWriter("data/%s_masked.avi" % timestamp, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10,
                      (WIDTH, HEIGHT))

channel = grpc.insecure_channel('localhost:50051', options=[
            ('grpc.max_send_message_length', MAX_MESSAGE_LENGTH),
            ('grpc.max_receive_message_length', MAX_MESSAGE_LENGTH),
        ])
stub = YOLOsvc_pb2_grpc.YOLOsvcStub(channel)

def detect_human(frame):
    x = base64.b64encode(frame)

    request = YOLOsvc_pb2.ImageB64(b64image=x, width=WIDTH, height=HEIGHT)

    results = stub.ObjectDetectionV2(request)

    # find out if a human was detected in the image
    # name column = "person", class = 0, confidence > .5, xmax, xmin, ymax, ymin
    human_detected = False
    human_coords = np.zeros(4)
    idx = 0
    for c in results.classes:
        if c == 0:
            human_detected = True
            human_coords = np.array([results.xmin[idx], results.ymin[idx], results.xmax[idx], results.ymax[idx]])

        idx += 1

    return human_detected, human_coords

frame_count = 0
while cap.isOpened():
    ret, frame = cap.read()

    if ret:
        frame_count += 1

        # for each frame, detect human, get ROI
        detected, human_coords = detect_human(frame)

        print("==> Frame %s, detected = %s, human_coords = %s" % (
            frame_count,
            detected,
            human_coords
        ))

        # for each frame, mask out off-ROI area.
        mask = np.zeros(frame.shape[:2], dtype="uint8")
        if detected:
            cv2.rectangle(mask, (int(human_coords[0]), int(human_coords[1])), (int(human_coords[2]), int(human_coords[3])), 255, -1)

        masked_frame = cv2.bitwise_and(frame, frame, mask=mask)

        # save masked frame to output video.
        out.write(masked_frame)
    else:
        cap.release()
        out.release()

out.release()
cap.release()
cv2.destroyAllWindows()

# verify that there are as many frames as there are actions, otherwise the transition planning module training will have issues
num_actions = 0
with open("data/%s.csv" % timestamp, 'r') as csvfile:
    datareader = csv.reader(csvfile, delimiter=',')
    for row in datareader:
        num_actions += 1

print("Loaded %s frames from video %s (there are %s corresponding actions)" % (
    frame_count,
    timestamp,
    num_actions
))
