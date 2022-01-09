import concurrent.futures as futures
import grpc
import logging
import YOLOsvc_pb2
import YOLOsvc_pb2_grpc
import torch
import numpy as np
import base64

MODEL_TYPE = 'person'  # for the hands model, replace this with 'hand'
MAX_MESSAGE_LENGTH = 6000000

if MODEL_TYPE == 'person':
    model = torch.hub.load('ultralytics/yolov5', 'yolov5l6')
elif MODEL_TYPE == 'hand':
    model = torch.hub.load('ultralytics/yolov5', 'custom', path='models/hands_model.pt', force_reload=True)  # default
else:
    print("Error: unknown model type. Use either 'person' or 'hand'")
    exit(-1)

model.conf = 0.75

def grpcImageToArray(img):
    x = []

    for row in img.img:
        px_list = []
        for px in row.row:
            rgb = []
            for rgb_val in px.px:
                rgb.append(rgb_val)

            px_list.append(rgb)

        x.append(px_list)

    return np.array(x, dtype='uint8')

def ResultsToDetections(results):

    detections_df = results.pandas().xyxy[0]
    detections_list = detections_df.values.tolist()

    xmin_array = []
    xmax_array = []
    ymin_array = []
    ymax_array = []
    classes_array = []
    confidence_array = []

    for d in detections_list:
        if d[-1] == MODEL_TYPE and d[-2] == 0:
            classes_array.append(d[-2])
            confidence_array.append(d[-3])
            xmin_array.append(d[0])
            xmax_array.append(d[2])
            ymin_array.append(d[1])
            ymax_array.append(d[3])

    return YOLOsvc_pb2.Detections( xmin=xmin_array,
                                   xmax=xmax_array,
                                   ymin=ymin_array,
                                   ymax=ymax_array,
                                   classes=classes_array,
                                   confidence=confidence_array)

class YOLOServicer(YOLOsvc_pb2_grpc.YOLOsvcServicer):
    """Provides methods that implement functionality of YOLOsvc server."""
    global model

    def __init__(self):
        pass

    def ObjectDetection(self, img, context):
        # from Image, get numpy array
        x = grpcImageToArray(img)

        results = model(x)

        detections = ResultsToDetections(results)
        return detections

    def ObjectDetectionV2(self, request, context):
        img = request.b64image.encode()

        tmp = base64.decodebytes(img)
        x = np.frombuffer(tmp, dtype='uint8').reshape(request.height, request.width, -1)

        results = model(x)

        detections = ResultsToDetections(results)
        return detections

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10), options=[
        ('grpc.max_send_message_length', MAX_MESSAGE_LENGTH),
        ('grpc.max_receive_message_length', MAX_MESSAGE_LENGTH)
    ])
    YOLOsvc_pb2_grpc.add_YOLOsvcServicer_to_server(
        YOLOServicer(), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    print("Service running...")
    server.wait_for_termination()


if __name__ == '__main__':
    logging.basicConfig()
    serve()