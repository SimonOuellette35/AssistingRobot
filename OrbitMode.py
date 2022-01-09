from Raspblock import Raspblock
import cv2
import grpc
import YOLOsvc_pb2
import YOLOsvc_pb2_grpc
import base64
import numpy as np
import time

MAX_MESSAGE_LENGTH = 6000000
WIDTH = 640
HEIGHT = 480

class OrbitMode:

    def __init__(self):
        self.robot = Raspblock()

        self.current_servo_pitch = 1000
        self.current_servo_yaw = 2500
        self.cam_delta = 50
        self.speed = 2

    def getObjectDetection(self, stub, img):

        x = base64.b64encode(img)

        request = YOLOsvc_pb2.ImageB64(b64image=x, width=WIDTH, height=HEIGHT)

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

    # TODO: a lot of this scan loop can be re-used for the main orbit mode (for the constant re-centering/re-detecting)
    def scan(self, scr):
        with grpc.insecure_channel('192.168.0.153:50051', options=[
            ('grpc.max_send_message_length', MAX_MESSAGE_LENGTH),
            ('grpc.max_receive_message_length', MAX_MESSAGE_LENGTH),
        ]) as channel:
            stub = YOLOsvc_pb2_grpc.YOLOsvcStub(channel)

            self.current_servo_yaw = 2500
            self.current_servo_pitch = 1500
            self.robot.Servo_control(self.current_servo_yaw, self.current_servo_pitch)

            cam = cv2.VideoCapture(0)
            cam.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
            cam.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)

            human_detected = False
            human_coords = None

            # Step 1: find human in image
            counter = 0
            while not human_detected:

                human_detected, human_coords = self.detect_human(stub, cam)

                # TODO: false positives are a problem... Try using higher quality YOLO model on the server side?
                #  Also make it an average detection over 5 frames? (across rotations so we don't waste time?)
                #  Maybe play with model conf on server side as well. And use joint probability over a few frames?

                if not human_detected:
                    scr.addstr(1, 0, "No human detected, moving camera to the right (%s)..." % counter)
                    scr.refresh()
                    if self.camRight():
                        # we reached the rightmost range of the servo.
                        # TODO: what next? spin right 90 degrees.

                        cam.release()
                        cv2.destroyAllWindows()

                        return

                    counter += 1

            # Step 2: fine tune to center the human: must see the top of the box as well (must not equal top of screen)
            screen_centerpoint = [WIDTH/2, HEIGHT/2]
            epsilon = 100

            scr.addstr(2, 0, "Phase 2: Centering the human...")
            scr.refresh()

            centered = False
            while not centered:
                box_centerpoint = [(human_coords[0] + human_coords[2]) / 2., (human_coords[1] + human_coords[3]) / 2.]
                box_width = (human_coords[2] - human_coords[0]) / float(WIDTH)

                scr.addstr(3, 0, "Not yet centered: centerpoint = %s, %s, width = %s" % (
                    box_centerpoint[0],
                    box_centerpoint[1],
                    box_width
                ))
                scr.refresh()

                centered = True

                # sub-task 1: center on the left-right axis
                if box_centerpoint[0] < screen_centerpoint[0] - epsilon:
                    self.camLeft()
                    centered = False
                elif box_centerpoint[0] > screen_centerpoint[0] + epsilon:
                    self.camRight()
                    centered = False

                # sub-task 2: find top of person (head)
                if human_coords[1] < 5:
                    self.camUp()
                    centered = False

                # sub-task 3: move forward/away so that the width of the detection box covers about X% of the screen width
                if box_width > 0.5:
                    self.backward()
                    centered = False
                elif box_width < 0.2:
                    self.forward()
                    centered = False

                # update human_coords (TODO: what if human no longer detected? Use object tracking for movement?)
                human_detected, human_coords = self.detect_human(stub, cam)

            cam.release()
            cv2.destroyAllWindows()

            # TODO: Step 3: do we see hands? face? Note: you can't always see the hands (visual obstructions, for example)

        # make a buzzer sound when human properly centered on
        self.robot.Buzzer_control(1)
        time.sleep(1)
        self.robot.Buzzer_control(0)

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
