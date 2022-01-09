from Raspblock import Raspblock
import cv2
import grpc
import YOLOsvc_pb2
import YOLOsvc_pb2_grpc
import base64
import numpy as np

MAX_MESSAGE_LENGTH = 6000000

class OrbitMode:

    def __init__(self, w, h):
        self.robot = Raspblock()

        self.w = w
        self.h = h
        self.current_servo_pitch = 2000
        self.current_servo_yaw = 500
        self.cam_delta = 50
        self.speed = 2

    def getObjectDetection(self, stub, img):

        x = base64.b64encode(img)

        request = YOLOsvc_pb2.ImageB64(b64image=x, width=self.w, height=self.h)

        return stub.ObjectDetectionV2(request)

    def detect_human(self, stub, cam):
        ret, img = cam.read()
        results = self.getObjectDetection(stub, img)

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

    def scan(self):
        with grpc.insecure_channel('localhost:50051', options=[
            ('grpc.max_send_message_length', MAX_MESSAGE_LENGTH),
            ('grpc.max_receive_message_length', MAX_MESSAGE_LENGTH),
        ]) as channel:
            stub = YOLOsvc_pb2_grpc.YOLOsvcStub(channel)

            self.current_servo_yaw = 500
            self.current_servo_pitch = 2000
            self.robot.Servo_control(self.current_servo_yaw, self.current_servo_pitch)

            cam = cv2.VideoCapture(0)
            cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 1024)

            human_detected = False
            human_coords = None

            # Step 1: find human in image
            while not human_detected:

                human_detected, human_coords = self.detect_human(stub, cam)

                if not human_detected:
                    if self.camRight():
                        # we reached the rightmost range of the servo.
                        # TODO: what next? spin right 90 degrees.

                        cam.release()
                        cv2.destroyAllWindows()

                        return

            # Step 2: fine tune to center the human: must see the top of the box as well (must not equal top of screen)
            screen_centerpoint = [640, 512]
            epsilon = 25

            centered = False
            while not centered:
                box_centerpoint = [(human_coords[0] + human_coords[2]) / 2., (human_coords[1] + human_coords[3]) / 2.]
                box_width = (human_coords[2] - human_coords[0]) / 1280.

                # sub-task 1: center on the left-right axis
                if box_centerpoint[0] < screen_centerpoint[0] - epsilon:
                    self.camLeft()
                elif box_centerpoint[0] > screen_centerpoint[0] + epsilon:
                    self.camRight()

                # sub-task 2: find top of person (head)
                if human_coords[1] < 5:
                    self.camUp()

                # sub-task 3: move forward/away so that the width of the detection box covers about X% of the screen width
                if box_width > 0.5:
                    self.backward()
                elif box_width < 0.2:
                    self.forward()

                # update human_coords (TODO: what if human no longer detected? Use object tracking for movement?)
                human_detected, human_coords = self.detect_human(stub, cam)

            cam.release()
            cv2.destroyAllWindows()

            # TODO: Step 3: do we see hands? face?

    def camRight(self):
        if self.current_servo_yaw >= 500 + self.cam_delta:
            self.current_servo_yaw -= self.cam_delta
            self.robot.Servo_control(self.current_servo_yaw, self.current_servo_pitch)
            return False
        else:
            return True

    def camLeft(self):
        if self.current_servo_yaw <= 2500 - self.cam_delta:
            self.current_servo_yaw += self.cam_delta
            self.robot.Servo_control(self.current_servo_yaw, self.current_servo_pitch)
            return False
        else:
            return True

    def camDown(self):
        if self.current_servo_pitch <= 2500 - self.cam_delta:
            self.current_servo_pitch += self.cam_delta
            self.robot.Servo_control(self.current_servo_yaw, self.current_servo_pitch)
            return False
        else:
            return True

    def camUp(self):
        if self.current_servo_pitch >= 500 + self.cam_delta:
            self.current_servo_pitch -= self.cam_delta
            self.robot.Servo_control(self.current_servo_yaw, self.current_servo_pitch)
            return False
        else:
            return True

    def counterClockwise(self):
        # TODO
        pass

    def clockwise(self):
        #TODO
        pass

    def forward(self):
        self.robot.Speed_Wheel_control(self.speed, self.speed, self.speed, self.speed)

    def backward(self):
        self.robot.Speed_Wheel_control(-self.speed, -self.speed, -self.speed, -self.speed)

    def increaseSpeed(self):
        if self.speed <= 23:
            self.speed += 2

    def decreaseSpeed(self):
        if self.speed >= 4:
            self.speed -= 2

    def close(self):
        del self.robot