from Raspblock import Raspblock
import grpc
import YOLOsvc_pb2
import YOLOsvc_pb2_grpc
import base64
import numpy as np
import time
from VideoGetter import VideoGetter

MAX_MESSAGE_LENGTH = 6000000
WIDTH = 640
HEIGHT = 480

class OrbitMode:

    def __init__(self):
        self.robot = Raspblock()

        self.current_servo_yaw = 2500
        self.speed = 8
        self.screen_centerpoint = [WIDTH / 2, HEIGHT / 2]
        self.epsilon = 25
        self.video_getter = VideoGetter(WIDTH, HEIGHT, 0).start()
        self.current_servo_pitch = 1250

    def getObjectDetection(self, stub, img):

        x = base64.b64encode(img)

        request = YOLOsvc_pb2.ImageB64(b64image=x, width=WIDTH, height=HEIGHT)

        return stub.ObjectDetectionV2(request)

    def detect_human(self, stub):

        img = None
        while img is None:
            img = self.video_getter.frame

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

    def phase1_detect(self, stub, scr):
        human_detected = False
        human_coords = None

        # Step 1: find human in image
        counter = 0
        spin_counter = 0
        while not human_detected:

            human_detected, human_coords = self.detect_human(stub)

            # TODO: false positives are a problem... Try using higher quality YOLO model on the server side?
            #  Also make it an average detection over 5 frames? (across rotations so we don't waste time?)
            #  Maybe play with model conf on server side as well. And use joint probability over a few frames?

            if not human_detected:
                scr.addstr(1, 0, "No human detected, moving camera to the right (%s)..." % counter)
                scr.refresh()
                if self.camRight(delta=100):
                    # we reached the rightmost range of the servo.
                    # spin right 180 degrees. Do this once, then give up.
                    spin_counter += 1

                    if spin_counter >= 2:
                        return None
                    else:
                        self.spinClockwise()
                        self.init_phase(self.current_servo_pitch)

                counter += 1

        return human_coords

    def phase2_center(self, stub, human_coords, scr):
        scr.addstr(2, 0, "Phase 2: Centering the human...")
        scr.refresh()

        good_frame = 0
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
            if box_centerpoint[0] < (self.screen_centerpoint[0] - self.epsilon):
                self.camLeft(delta=50)
                centered = False

                # TODO: handle the case where the camera is already at the leftmost rotation. Must rotate the car itself.

                scr.addstr(4, 0, "==> Off to the LEFT by %s pixels (box center = %s, THRESHOLD = %s)" % (
                    box_centerpoint[0] - self.screen_centerpoint[0],
                    box_centerpoint[0],
                    (self.screen_centerpoint[0] - self.epsilon)
                ))
                scr.refresh()

            elif box_centerpoint[0] > self.screen_centerpoint[0] + self.epsilon:
                self.camRight(delta=50)
                centered = False

                # TODO: handle the case where the camera is already at the rightmost rotation. Must rotate the car itself.
                scr.addstr(4, 0, "==> Off to the RIGHT by %s pixels (box center = %s, THRESHOLD = %s)" % (
                    self.screen_centerpoint[0] - box_centerpoint[0],
                    box_centerpoint[0],
                    (self.screen_centerpoint[0] + self.epsilon)
                ))
                scr.refresh()

            # sub-task 2: find top of person (head)
            elif human_coords[1] < 1.:
                self.camUp(delta=40)
                centered = False

                scr.addstr(4, 0, "==> TOO LOW, top pixel is %s" % (
                    human_coords[1]
                ))
                scr.refresh()

            # sub-task 3: move forward/away so that the width of the detection box covers about X% of the screen width
            elif box_width > 0.5:
                self.backward()
                centered = False

                scr.addstr(4, 0, "==> TOO WIDE")
                scr.refresh()
            elif box_width < 0.2:
                self.forward()
                centered = False

                scr.addstr(4, 0, "==> TOO NARROW")
                scr.refresh()

            if centered:
                good_frame += 1

                if good_frame < 4:
                    centered = False

            # update human_coords
            human_detected, human_coords = self.detect_human(stub)

        scr.addstr(4, 0, "==> SUCCESS !")
        scr.refresh()

        return centered

    def init_phase(self, init_pitch=1250):
        self.current_servo_yaw = 2500
        self.current_servo_pitch = init_pitch
        self.robot.Servo_control(self.current_servo_yaw, self.current_servo_pitch)

    # TODO: a lot of this scan loop can be re-used for the main orbit mode (for the constant re-centering/re-detecting)
    def scan(self, scr):
        with grpc.insecure_channel('192.168.0.153:50051', options=[
            ('grpc.max_send_message_length', MAX_MESSAGE_LENGTH),
            ('grpc.max_receive_message_length', MAX_MESSAGE_LENGTH),
        ]) as channel:
            stub = YOLOsvc_pb2_grpc.YOLOsvcStub(channel)

            init_pitch = 1250
            self.init_phase()

            centered = False
            while not centered:

                # Phase 1: scan the room with large camera and robot movements to identify a human candidate
                human_coords = self.phase1_detect(stub, scr)

                if human_coords is None:
                    scr.addstr(0, 0, "ERROR ==> Failed to detect human in first scan phase! Trying again...")
                    scr.refresh()

                    # Start over, looking higher
                    init_pitch -= 250
                    if init_pitch <= 500:
                        init_pitch = 600

                    self.init_phase(init_pitch)
                else:
                    # Step 2: fine tune to center the human: must see the top of the box as well (must not equal top of screen)
                    centered = self.phase2_center(stub, human_coords, scr)

            self.video_getter.stop()

            # TODO: Step 3: do we see hands? face? Note: you can't always see the hands (visual obstructions, for example)

        # make a buzzer sound when human properly centered on
        self.robot.Buzzer_control(1)
        time.sleep(1)
        self.robot.Buzzer_control(0)

    def camRight(self, delta=50):
        if self.current_servo_yaw >= 500 + delta:
            self.current_servo_yaw -= delta
            self.robot.Servo_control(self.current_servo_yaw, self.current_servo_pitch)
            return False
        else:
            return True

    def camLeft(self, delta=50):
        if self.current_servo_yaw <= 2500 - delta:
            self.current_servo_yaw += delta
            self.robot.Servo_control(self.current_servo_yaw, self.current_servo_pitch)
            return False
        else:
            return True

    def camDown(self, delta=50):
        if self.current_servo_pitch <= 2500 - delta:
            self.current_servo_pitch += delta
            self.robot.Servo_control(self.current_servo_yaw, self.current_servo_pitch)
            return False
        else:
            return True

    def camUp(self, delta=50):
        if self.current_servo_pitch >= 500 + delta:
            self.current_servo_pitch -= delta
            self.robot.Servo_control(self.current_servo_yaw, self.current_servo_pitch)
            return False
        else:
            return True

    def orbitCounterClockwise(self):
        # TODO
        pass

    def orbitClockwise(self):
        #TODO
        pass

    def spinClockwise(self):
        for _ in range(10):
            self.robot.Speed_Wheel_control(-10, -10, 10, 10)
            time.sleep(0.1)

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
