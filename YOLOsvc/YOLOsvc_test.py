from __future__ import print_function

import logging
import random
import grpc
import YOLOsvc_pb2
import YOLOsvc_pb2_grpc
import cv2
import time

IMAGE_FILENAME = "../data/person.jpg"
MAX_MESSAGE_LENGTH = 6000000

def arrayTogRPCImage(img):

    output = []
    for x in img:
        pixel_row = []
        for y in x:
            rgb_vals = []
            for z in y:
                rgb_vals.append(z)

            rgb = YOLOsvc_pb2.RGB(px=rgb_vals)
            pixel_row.append(rgb)

        row = YOLOsvc_pb2.PixelRow(row=pixel_row)
        output.append(row)

    return YOLOsvc_pb2.Image(img=output)

def getObjectDetection(stub):

    # load image from file
    img = cv2.imread(IMAGE_FILENAME)

    # transform to YOLOsvc Image format
    x = arrayTogRPCImage(img)
    return stub.ObjectDetection(x)

def run():
    # NOTE(gRPC Python Team): .close() is possible on a channel and should be
    # used in circumstances in which the with statement does not fit the needs
    # of the code.
    with grpc.insecure_channel('localhost:50051', options=[
        ('grpc.max_send_message_length', MAX_MESSAGE_LENGTH),
        ('grpc.max_receive_message_length', MAX_MESSAGE_LENGTH),
    ]) as channel:
        stub = YOLOsvc_pb2_grpc.YOLOsvcStub(channel)

        print("-------------- Object Detection --------------")
        start_time = time.time()
        results = getObjectDetection(stub)
        end_time = time.time()

        print ("==> Object detection call took %s seconds." % (end_time - start_time))
        # process and display results
        print("Detected classes: ", results.classes)
        
if __name__ == '__main__':
    logging.basicConfig()
    run()
