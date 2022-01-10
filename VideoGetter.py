import cv2
from threading import Thread

class VideoGetter:
    """
    Class that continuously gets frames from a VideoCapture object
    with a dedicated thread.
    """

    def __init__(self, WIDTH=640, HEIGHT=480, src=0):
        self.stream = cv2.VideoCapture(src)
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)

        (self.grabbed, self.frame) = self.stream.read()
        self.stopped = False


    def start(self):
        Thread(target=self.get, args=()).start()
        return self

    def get(self):
        while not self.stopped:
            if not self.grabbed:
                self.stop()
            else:
                (self.grabbed, self.frame) = self.stream.read()

        self.stream.release()
        cv2.destroyAllWindows()

    def stop(self):
        self.stopped = True
