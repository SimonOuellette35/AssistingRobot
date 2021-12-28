# AssistingRobot
Thesis project: motion planning for active human activity recognition

**Physical setup:**
* Currently using a Yahboom Raspblock robot (http://www.yahboom.net/study/Raspblock)
* The code runs on a Raspberry Pi 4

**Files:**
* DataCaptureRaw.py: uses keyboard capture to control the robot, and saves a .csv file in which each row corresponds to the actions taken by the robot before the corresponding frame in the video. Each action row is a 6-element vector representing: (yaw of camera servo, pitch of camera servo, wheel #1 speed, wheel #2 speed, wheel#3 speed, wheel #4 speed)

* DataCaptureOrbit.py[work in progress]: uses orbit mode (OrbitMode.py) to control the robot for data capture. See details of orbit mode below.

* run_YOLOcam.py: runs test code that loads a YOLO model and, running on a live camera, shows identified objects in the video stream in real-time.

**Orbit mode:**

In raw mode, the control keys translate directly into raw robot movements. For example, the "a" key corresponds to left movement, and results in the robot sliding to the left by an amount determined by the speed parameter.

In orbit mode, the movement space is transformed: it becomes an orbit about the detected human. It leverages object detection (YoloV5) to detect the human and apply the necessary motor and cam servo adjustments to keep the human centered in the frame. When the user pushes the "a" key, now the corresponding robot movement becomes a clockwise motion in the orbit around the human. The "d" key becomes a counter-clockwise movement in the same orbit. The "w" key results in moving closer to the person (reducing the orbit circumference). The "s" key moves away from the person (increasing the orbit circumference).

**YOLO models:**

Currently I am using two models:

1 - the yolov5s pretrained model "as is" to detect persons in the image. 

2 - a hand detection model (models/hands_model.pt) which I trained by first loading a pretrained yolov5s model, and then I finetuned it (over 10 epochs of training) on the oxford hands dataset. The annotations for the dataset were generated automatically using https://github.com/SignusRobotics/oxford-hands-to-yolo. The current model has a MAP score of around 82%: there is plenty of room for improvement.
