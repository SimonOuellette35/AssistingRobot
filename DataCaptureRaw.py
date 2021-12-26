import curses
from curses import wrapper
from Raspblock import Raspblock
import cv2
from datetime import datetime
import numpy as np
import csv

robot = Raspblock()

current_servo_pitch = 1500
current_servo_yaw = 1500
cam_delta = 50
speed = 2
cap = None
out = None
frame_actions = []
last_action = np.zeros(6) # (servo yaw, servo pitch, wheel #1, wheel #2, wheel #3, wheel #4)
actions_file = None
datawriter = None

# init servo camera:
robot.Servo_control(current_servo_yaw, current_servo_pitch)

def getTimestampFilename():
    now = datetime.now()
    basename = now.strftime("%Y-%m-%d_%H-%M-%S")

    return "data/%s" % basename

def main(scr):
    global current_servo_pitch
    global current_servo_yaw
    global speed
    global cap
    global out
    global frame_actions
    global last_action
    global actions_file
    global datawriter

    scr.clear()

    while True:
        if cap is not None and out is not None:
            # Need to capture a video frame!
            ret, frame = cap.read()

            if ret:
                out.write(frame)

                # save last_action to file
                datawriter.writerow(last_action)

        last_action = np.zeros(6)

        c = scr.getch()

        if c == ord(' '):
            return

        # Control the camera
        if c == curses.KEY_RIGHT:
            if current_servo_yaw >= 500 + cam_delta:
                current_servo_yaw -= cam_delta
                robot.Servo_control(current_servo_yaw, current_servo_pitch)
                last_action[0] = -cam_delta

        if c == curses.KEY_LEFT:
            if current_servo_yaw <= 2500 - cam_delta:
                current_servo_yaw += cam_delta
                robot.Servo_control(current_servo_yaw, current_servo_pitch)
                last_action[0] = cam_delta

        if c == curses.KEY_DOWN:
            if current_servo_pitch <= 2500 - cam_delta:
                current_servo_pitch += cam_delta
                robot.Servo_control(current_servo_yaw, current_servo_pitch)
                last_action[1] = cam_delta

        if c == curses.KEY_UP:
            if current_servo_pitch >= 500 + cam_delta:
                current_servo_pitch -= cam_delta
                robot.Servo_control(current_servo_yaw, current_servo_pitch)
                last_action[1] = -cam_delta

        # Control the motor
        if c == ord('a'):   # left strafe
            robot.Speed_Wheel_control(speed, -speed, speed, -speed)
            last_action[2] = speed
            last_action[3] = -speed
            last_action[4] = speed
            last_action[5] = -speed
        if c == ord('d'):   # right strafe
            robot.Speed_Wheel_control(-speed, speed, -speed, speed)
            last_action[2] = -speed
            last_action[3] = speed
            last_action[4] = -speed
            last_action[5] = speed
        if c == ord('w'):   # forward
            robot.Speed_Wheel_control(speed, speed, speed, speed)
            last_action[2] = speed
            last_action[3] = speed
            last_action[4] = speed
            last_action[5] = speed
        if c == ord('s'):   # backward
            robot.Speed_Wheel_control(-speed, -speed, -speed, -speed)
            last_action[2] = -speed
            last_action[3] = -speed
            last_action[4] = -speed
            last_action[5] = -speed
        if c == ord('e'):   # spin right
            robot.Speed_Wheel_control(-speed, -speed, speed, speed)
            last_action[2] = -speed
            last_action[3] = -speed
            last_action[4] = speed
            last_action[5] = speed
        if c == ord('q'):   # spin left
            robot.Speed_Wheel_control(speed, speed, -speed, -speed)
            last_action[2] = speed
            last_action[3] = speed
            last_action[4] = -speed
            last_action[5] = -speed

        # increase/decrease motor speed
        if c == curses.KEY_PPAGE:   # Page up: increase speed
            if speed <= 23:
                speed += 2

        if c == curses.KEY_NPAGE:   # Page down: decrease speed
            if speed >= 4:
                speed -= 2

        if c == ord('f'):   # Start filming
            if cap is None:
                filename = getTimestampFilename()
                cap = cv2.VideoCapture(0)

                scr.addstr(0, 0, "Recording video to file: %s.avi" % filename)

                frame_width = int(cap.get(3))
                frame_height = int(cap.get(4))

                out = cv2.VideoWriter("%s.avi" % filename, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10,
                                      (frame_width, frame_height))

                # open actions file
                actions_file = open("%s.csv" % filename, 'w')
                datawriter = csv.writer(actions_file, delimiter=',')


        if c == curses.KEY_BACKSPACE:   # Stop filming
            if cap is not None:
                cap.release()
                out.release()
                cap = None
                out = None

                # close actions file
                actions_file.close()
                actions_file = None

                scr.addstr(0, 0, "Done recoding. Save data to file....                         ")

wrapper(main)
del robot