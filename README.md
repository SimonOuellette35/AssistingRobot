# AssistingRobot
Thesis project: motion planning for active human activity recognition

Physical setup:
* Currently using a Yahboom Raspblock robot (http://www.yahboom.net/study/Raspblock)
* The code runs on a Raspberry Pi 4

Files:
* DataCaptureRaw.py: uses keyboard capture to control the robot, and saves a .csv file in which each row corresponds to the actions taken by the robot before the corresponding frame in the video. Each action row is a 6-element vector representing: (yaw of camera servo, pitch of camera servo, wheel #1 speed, wheel #2 speed, wheel#3 speed, wheel #4 speed)

* DataCaptureOrbit.py[work in progress]: uses orbit mode (OrbitMode.py) to control the robot for data capture. See details of orbit mode below.

Orbit mode:

In raw mode, the control keys translate directly into raw robot movements. For example, the "a" key corresponds to left movement, and results in the robot sliding to the left by an amount determined by the speed parameter.

In orbit mode, the movement space is transformed: it becomes an orbit about the detected human. It leverages object detection (YoloV5) to detect the human and apply the necessary motor and cam servo adjustments to keep the human centered in the frame. When the user pushes the "a" key, now the corresponding robot movement becomes a clockwise motion in the orbit around the human. The "d" key becomes a counter-clockwise movement in the same orbit. The "w" key results in moving closer to the person (reducing the orbit circumference). The "s" key moves away from the person (increasing the orbit circumference).
