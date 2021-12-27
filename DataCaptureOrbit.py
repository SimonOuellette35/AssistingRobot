import curses
from curses import wrapper
import cv2
from datetime import datetime
import numpy as np
import csv
from OrbitMode import OrbitMode

# This data capture script works in orbit mode. In orbit mode, object detection is used to detect the human
#  and keep them centered in a the video frame. Left and right transition are no longer treated as raw left-right
#  transitions, but instead become orbital movements in clockwise and anti-clockwise motion around the human subject.
#  Orbit mode serves several purposes:
#  1. Makes controlling the robot for data capture much easier
#  2. Abstracts away the lower-level motion planning involved in keeping the human centered in the frame
#  3. Abstracts away the hardware specific commands, allowing to re-use the same higher-level ML/planning algo on
#     different robots.

cap = None
out = None
frame_actions = []
last_action = np.zeros(6) # (servo yaw, servo pitch, wheel #1, wheel #2, wheel #3, wheel #4)
actions_file = None
datawriter = None

orbit = OrbitMode()

def getTimestampFilename():
    now = datetime.now()
    basename = now.strftime("%Y-%m-%d_%H-%M-%S")

    return "data/%s" % basename

def main(scr):
    global cap
    global out
    global frame_actions
    global last_action
    global actions_file
    global datawriter

    scr.clear()

    # init servo camera:
    scr.addstr(0, 0, "Scanning for human...")
    orbit.scan()
    scr.addstr(1, 0, "Found human!")

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

        # # Control the camera
        # if c == curses.KEY_RIGHT:
        #     orbit.camRight()
        #     last_action[0] = -cam_delta
        #
        # elif c == curses.KEY_LEFT:
        #     orbit.camLeft()
        #     last_action[0] = cam_delta
        #
        # elif c == curses.KEY_DOWN:
        #     last_action[1] = cam_delta
        #
        # elif c == curses.KEY_UP:
        #     last_action[1] = -cam_delta
        #
        # # Control the motor
        # elif c == ord('a'):   # left strafe
        #     orbit.clockwise()
        #     last_action[2] = speed
        #     last_action[3] = -speed
        #     last_action[4] = speed
        #     last_action[5] = -speed
        # elif c == ord('d'):   # right strafe
        #     orbit.counterClockwise()
        #     last_action[2] = -speed
        #     last_action[3] = speed
        #     last_action[4] = -speed
        #     last_action[5] = speed
        # elif c == ord('w'):   # forward
        #     orbit.forward()
        #     last_action[2] = speed
        #     last_action[3] = speed
        #     last_action[4] = speed
        #     last_action[5] = speed
        # elif c == ord('s'):   # backward
        #     orbit.backward()
        #     last_action[2] = -speed
        #     last_action[3] = -speed
        #     last_action[4] = -speed
        #     last_action[5] = -speed

        # increase/decrease motor speed
        if c == curses.KEY_PPAGE:   # Page up: increase speed
            orbit.increaseSpeed()

        if c == curses.KEY_NPAGE:   # Page down: decrease speed
            orbit.decreaseSpeed()

        if c == ord('f'):   # Start filming
            if cap is None:
                filename = getTimestampFilename()
                cap = cv2.VideoCapture(0)

                scr.addstr(2, 0, "Recording video to file: %s.avi" % filename)

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

                scr.addstr(3, 0, "Done recoding. Saved data to file....                         ")

wrapper(main)
orbit.close()
