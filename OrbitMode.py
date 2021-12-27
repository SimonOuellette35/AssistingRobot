from Raspblock import Raspblock
import cv2
import torch
import numpy as np

class OrbitMode:

    def __init__(self):
        self.robot = Raspblock()

        self.current_servo_pitch = 2000
        self.current_servo_yaw = 500
        self.cam_delta = 50
        self.speed = 2
        self.yolo = torch.hub.load('ultralytics/yolov5', 'yolov5s6')
        self.yolo.conf = 0.5

    def detect_human(self, cam):
        ret, img = cam.read()
        results = self.yolo(img)

        # find out if a human was detected in the image
        # name column = "person", class = 0, confidence > .5, xmax, xmin, ymax, ymin
        detections_df = results.pandas().xyxy[0]
        detections_list = detections_df.values.tolist()

        human_detected = False
        for d in detections_list:
            if d[-1] == 'person' and d[-2] == 0:
                human_detected = True
                human_coords = np.array(d[:4])

        return human_detected, human_coords

    def scan(self):
        self.current_servo_yaw = 500
        self.current_servo_pitch = 2000
        self.robot.Servo_control(self.current_servo_yaw, self.current_servo_pitch)

        # TODO: find human in img. Generate movement adjustments and return when ok.
        cam = cv2.VideoCapture(0)
        cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 1024)

        human_detected = False
        human_coords = None

        # Step 1: find human in image
        while not human_detected:

            human_detected, human_coords = self.detect_human(cam)

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
            human_detected, human_coords = self.detect_human(cam)

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