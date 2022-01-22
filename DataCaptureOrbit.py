import curses
from curses import wrapper
from datetime import datetime
import numpy as np
import csv
from OrbitMode import OrbitMode
import cv2

# This data capture script works in orbit mode. In orbit mode, object detection is used to detect the human
#  and keep them centered in a the video frame. Left and right transition are no longer treated as raw left-right
#  transitions, but instead become orbital movements in clockwise and anti-clockwise motion around the human subject.
#  Orbit mode serves several purposes:
#  1. Makes controlling the robot for data capture much easier
#  2. Abstracts away the lower-level motion planning involved in keeping the human centered in the frame
#  3. Abstracts away the hardware specific commands, allowing to re-use the same higher-level ML/planning algo on
#     different robots.

out = None
frame_actions = []
last_action = 0.
actions_file = None
datawriter = None

WIDTH = 640
HEIGHT = 480

def getTimestampFilename():
    now = datetime.now()
    basename = now.strftime("%Y-%m-%d_%H-%M-%S")

    return "data/%s" % basename

def main(scr):
    global out
    global frame_actions
    global last_action
    global actions_file
    global datawriter

    scr.clear()

    orbit = OrbitMode(scr, WIDTH, HEIGHT)

    # init servo camera:
    while True:
        if out is not None:
            frame = orbit.video_getter.frame

            # Record full frame. A separate tool will load the frames, use the same object detector to detect
            #  human_coords, and use them and raw images to produce the masked dataset, which will probably be used for the training.

            out.write(frame)

            # save last_action to file
            datawriter.writerow([last_action])

        last_action = 0.

        c = scr.getch()

        if c == ord(' '):
            if out is not None:
                out.release()
                out = None

                # close actions file
                actions_file.close()
                actions_file = None

                scr.addstr(3, 0, "Done recoding. Saved data to file....                         ")

            orbit.close()
            return

        # Control the motor
        elif c == ord('a'):   # clockwise orbit
            orbit.orbitClockwise()
            last_action = 1
        elif c == ord('d'):   # counter-clockwise orbit
            orbit.orbitCounterClockwise()
            last_action = 2
        elif c == ord('w'):   # forward
            orbit.forward(delta=10)
            last_action = 3
        elif c == ord('s'):   # backward
            orbit.backward(delta=10)
            last_action = 4

        if c == ord('i'):
            scr.addstr(0, 0, "Initializing scanner...")
            orbit.scan(scr)

        if c == ord('f'):   # Start filming
            if out is None:
                filename = getTimestampFilename()

                scr.addstr(2, 0, "Recording video to file: %s.avi" % filename)

                out = cv2.VideoWriter("%s.avi" % filename, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10,
                                      (WIDTH, HEIGHT))

                # open actions file
                actions_file = open("%s.csv" % filename, 'w')
                datawriter = csv.writer(actions_file, delimiter=',')

        if c == curses.KEY_BACKSPACE:   # Stop filming
            if out is not None:
                out.release()
                out = None

                # close actions file
                actions_file.close()
                actions_file = None

                scr.addstr(3, 0, "Done recoding. Saved data to file....                         ")

wrapper(main)
