from Raspblock import Raspblock
import grpc
import YOLOsvc_pb2
import YOLOsvc_pb2_grpc
import base64
import numpy as np
import time
from VideoGetter import VideoGetter

MAX_MESSAGE_LENGTH = 6000000

class OrbitMode:

    def __init__(self, scr, width, height):
        self.robot = Raspblock()
        self.scr = scr
        self.current_servo_yaw = 2500
        self.speed = 8
        self.WIDTH = width
        self.HEIGHT = height
        self.screen_centerpoint = [width / 2, height / 2]
        self.epsilon = 25
        self.video_getter = VideoGetter(width, height, 0).start()
        self.current_servo_pitch = 1250
        self.channel = grpc.insecure_channel('192.168.0.193:50051', options=[
            ('grpc.max_send_message_length', MAX_MESSAGE_LENGTH),
            ('grpc.max_receive_message_length', MAX_MESSAGE_LENGTH),
        ])
        self.stub = YOLOsvc_pb2_grpc.YOLOsvcStub(self.channel)

    def getObjectDetection(self, img):

        x = base64.b64encode(img)

        request = YOLOsvc_pb2.ImageB64(b64image=x, width=self.WIDTH, height=self.HEIGHT)

        return self.stub.ObjectDetectionV2(request)

    def detect_human(self):

        img = None
        while img is None:
            img = self.video_getter.frame

        results = self.getObjectDetection(img)

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

    def phase1_detect(self):
        human_detected = False
        human_coords = None

        # Step 1: find human in image
        counter = 0
        spin_counter = 0
        while not human_detected:

            human_detected, human_coords = self.detect_human()

            # TODO: false positives are a problem... Try using higher quality YOLO model on the server side?
            #  Also make it an average detection over 5 frames? (across rotations so we don't waste time?)
            #  Maybe play with model conf on server side as well. And use joint probability over a few frames?

            # TODO: review literature on increasing detection confidence across frames. This could be area of
            #  improvement for a paper?

            if not human_detected:
                self.scr.addstr(1, 0, "No human detected, moving camera to the right (%s)..." % counter)
                self.scr.refresh()
                if self.camRight(delta=100):
                    # we reached the rightmost range of the servo.
                    # spin right 180 degrees. Do this once, then give up.
                    spin_counter += 1

                    if spin_counter >= 2:
                        return None
                    else:
                        self.spin180Clockwise()
                        self.init_phase(self.current_servo_pitch)

                counter += 1

        return human_coords

    def phase2_center(self, human_coords):
        self.scr.addstr(2, 0, "Phase 2: Centering the human...")
        self.scr.refresh()

        good_frame = 0
        centered = False
        while not centered:
            box_centerpoint = [(human_coords[0] + human_coords[2]) / 2., (human_coords[1] + human_coords[3]) / 2.]
            box_width = (human_coords[2] - human_coords[0]) / float(self.WIDTH)

            self.scr.addstr(3, 0, "Not yet centered: centerpoint = %s, %s, width = %s" % (
                box_centerpoint[0],
                box_centerpoint[1],
                box_width
            ))
            self.scr.refresh()

            centered = True

            # sub-task 1: center on the left-right axis
            if box_centerpoint[0] < (self.screen_centerpoint[0] - self.epsilon):
                limit_reached = self.camLeft(delta=50)
                centered = False

                # handle the case where the camera is already at the leftmost rotation. Must rotate the car itself.
                if limit_reached:
                    self.spin90CounterClockwise()
                    time.sleep(0.2)
                    self.camRight(delta=250)

                self.scr.addstr(4, 0, "==> Off to the LEFT by %s pixels (box center = %s, THRESHOLD = %s)" % (
                    box_centerpoint[0] - self.screen_centerpoint[0],
                    box_centerpoint[0],
                    (self.screen_centerpoint[0] - self.epsilon)
                ))
                self.scr.refresh()

            elif box_centerpoint[0] > self.screen_centerpoint[0] + self.epsilon:
                limit_reached = self.camRight(delta=50)
                centered = False

                # handle the case where the camera is already at the rightmost rotation. Must rotate the car itself.
                if limit_reached:
                    self.spin90Clockwise()
                    self.camLeft(delta=250)

                self.scr.addstr(4, 0, "==> Off to the RIGHT by %s pixels (box center = %s, THRESHOLD = %s)" % (
                    self.screen_centerpoint[0] - box_centerpoint[0],
                    box_centerpoint[0],
                    (self.screen_centerpoint[0] + self.epsilon)
                ))
                self.scr.refresh()

            # sub-task 2: find top of person (head)
            elif human_coords[1] < 1.:
                self.camUp(delta=40)
                centered = False

                self.scr.addstr(4, 0, "==> TOO LOW, top pixel is %s" % (
                    human_coords[1]
                ))
                self.scr.refresh()

            # sub-task 3: move forward/away so that the width of the detection box covers about X% of the screen width
            elif box_width > 0.5:
                self.backward()
                centered = False

                self.scr.addstr(4, 0, "==> TOO WIDE")
                self.scr.refresh()
            elif box_width < 0.2:
                self.forward(delta=10)
                centered = False

                self.scr.addstr(4, 0, "==> TOO NARROW")
                self.scr.refresh()

            if centered:
                good_frame += 1

                if good_frame < 1:      # Increase this number to number of consecutive frames of detection to be considered centered.
                    centered = False

            # update human_coords
            human_detected, human_coords = self.detect_human()

        self.scr.addstr(4, 0, "==> SUCCESS !")
        self.scr.refresh()

        return centered

    def close(self):
        self.video_getter.stop()
        self.channel.close()
        del self.robot

    # this function re-orients the car so that it faces the locked-in human, repositioning the camera so that
    # it is facing forward.
    def lock_in(self):
        # 1000 = right-most cam position, -1000 = left-most cam position
        cam_position = 1500 - self.current_servo_yaw

        self.scr.addstr(6, 0, "==> Lock in: current yaw is %s, cam_position = %s" % (self.current_servo_yaw, cam_position))
        self.scr.refresh()

        if abs(cam_position) < 10:  # already locked in
            return

        # Function that maps a cam_position to a corresponding car body spin rotation
        # Reference points: -1000 means spin left 90 degrees. 1000 means spin right 90 degrees. 0 means don't move.
        delta = (cam_position / 1000) * 30

        if delta < 0:
            self.spinLeft(abs(delta))
        elif delta > 0:
            self.spinRight(delta)

        time.sleep(0.2)

        # Position camera to center of yaw axis
        self.current_servo_yaw = 1500
        self.robot.Servo_control(self.current_servo_yaw, self.current_servo_pitch)

        time.sleep(0.5)

    def init_phase(self, init_pitch=1250):
        self.current_servo_yaw = 2500
        self.current_servo_pitch = init_pitch
        self.robot.Servo_control(self.current_servo_yaw, self.current_servo_pitch)

    def scan(self, scr):
        init_pitch = 1250
        self.init_phase()

        centered = False
        while not centered:

            # Phase 1: scan the room with large camera and robot movements to identify a human candidate
            human_coords = self.phase1_detect()

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
                centered = self.phase2_center(human_coords)

        #self.video_getter.stop()

        # TODO: Step 3: do we see hands? face? Note: you can't always see the hands (visual obstructions, for example)

        self.lock_in()

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

    def strafeRight(self, delta=50):
        if delta <= 20:
            self.robot.Speed_Wheel_control(int(-delta), int(delta), int(-delta), int(delta))
        else:
            iterations = int(delta // 10)
            mini_delta = int(delta / iterations)

            for _ in range(iterations):
                self.robot.Speed_Wheel_control(int(-mini_delta), int(mini_delta), int(-mini_delta), int(mini_delta))
                time.sleep(0.2)

    def strafeLeft(self, delta=50):
        if delta <= 20:
            self.robot.Speed_Wheel_control(int(delta), int(-delta), int(delta), int(-delta))
        else:
            iterations = int(delta // 10)
            mini_delta = int(delta / iterations)

            for _ in range(iterations):
                self.robot.Speed_Wheel_control(int(mini_delta), int(-mini_delta), int(mini_delta), int(-mini_delta))
                time.sleep(0.2)

    def orbitCounterClockwise(self, delta=50):
        # Strafe right
        self.strafeRight(delta)

        # Re-center to human
        detected, human_coords = self.detect_human()

        # TODO: if the box width changed significantly, might need to move forward a bit to adjust to orbit

        if detected:
            self.phase2_center(human_coords)
            self.lock_in()
        else:
            # TODO: must resume scanning, in the right direction (starting from current cam position)...
            pass

    def orbitClockwise(self, delta=50):
        # Strafe right
        self.strafeLeft(delta)

        # Re-center to human
        detected, human_coords = self.detect_human()

        # TODO: if the box width changed significantly, might need to move forward a bit to adjust to orbit

        if detected:
            self.phase2_center(human_coords)
            self.lock_in()
        else:
            # TODO: must resume scanning, in the right direction (starting from current cam position)...
            pass

    def spin180Clockwise(self):
        for _ in range(10):
            self.robot.Speed_Wheel_control(-10, -10, 10, 10)
            time.sleep(0.1)

    def spin90Clockwise(self):
        for _ in range(4):
            self.robot.Speed_Wheel_control(-10, -10, 10, 10)
            time.sleep(0.1)

    def spin90CounterClockwise(self):
        for _ in range(4):
            self.robot.Speed_Wheel_control(10, 10, -10, -10)
            time.sleep(0.1)

    def spinLeft(self, delta):
        self.robot.Speed_Wheel_control(int(delta), int(delta), int(-delta), int(-delta))

    def spinRight(self, delta):
        if delta <= 15:
            self.robot.Speed_Wheel_control(int(-delta), int(-delta), int(delta), int(delta))
        else:
            iterations = int(delta // 10)
            mini_delta = int(delta / iterations)

            for _ in range(iterations):
                self.robot.Speed_Wheel_control(int(-mini_delta), int(-mini_delta), int(mini_delta), int(mini_delta))
                time.sleep(0.2)

    def forward(self, delta):
        self.robot.Speed_Wheel_control(int(delta), int(delta), int(delta), int(delta))

    def backward(self, delta):
        self.robot.Speed_Wheel_control(-int(delta), -int(delta), -int(delta), -int(delta))

    def increaseSpeed(self):
        if self.speed <= 23:
            self.speed += 2

    def decreaseSpeed(self):
        if self.speed >= 4:
            self.speed -= 2
