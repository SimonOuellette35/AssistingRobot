from Raspblock import Raspblock

class OrbitMode:

    def __init__(self):
        self.robot = Raspblock()

        self.current_servo_pitch = 1500
        self.current_servo_yaw = 1500
        self.cam_delta = 50
        self.speed = 2

    def init(self):
        self.robot.Servo_control(1500, 1500)

    def camRight(self):
        if self.current_servo_yaw >= 500 + self.cam_delta:
            self.current_servo_yaw -= self.cam_delta
            self.robot.Servo_control(self.current_servo_yaw, self.current_servo_pitch)

    def camLeft(self):
        if self.current_servo_yaw <= 2500 - self.cam_delta:
            self.current_servo_yaw += self.cam_delta
            self.robot.Servo_control(self.current_servo_yaw, self.current_servo_pitch)

    def camDown(self):
        if self.current_servo_pitch <= 2500 - self.cam_delta:
            self.current_servo_pitch += self.cam_delta
            self.robot.Servo_control(self.current_servo_yaw, self.current_servo_pitch)

    def camUp(self):
        if self.current_servo_pitch >= 500 + self.cam_delta:
            self.current_servo_pitch -= self.cam_delta
            self.robot.Servo_control(self.current_servo_yaw, self.current_servo_pitch)

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