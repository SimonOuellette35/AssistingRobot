# AssistingRobot
Thesis project: motion planning for active human activity recognition

Physical setup:
* Currently using a Yahboom Raspblock robot (http://www.yahboom.net/study/Raspblock)
* The code runs on a Raspberry Pi 4

Files:
* DataCaptureRaw.py: uses keyboard capture to control the robot, and saves a .csv file in which each row corresponds to the actions taken by the robot before the corresponding frame in the video. Each action row is a 6-element vector representing: (yaw of camera servo, pitch of camera servo, wheel #1 speed, wheel #2 speed, wheel#3 speed, wheel #4 speed)
