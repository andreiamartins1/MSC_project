import cv2
import threading
import time
import numpy as np


lock = threading.Lock()

class Characterizer(object):


    def __init__(self, fps=25, shape=(256, 256, 3), prediction_timeout=10) -> None:
        self.fps = fps
        self.shape = shape
        self.prediction_timeout = prediction_timeout

        self.frame = None
        self.prediction = None
        self.average_prediction = None
        self.prediction_count = 0

        self.new_frame = False
        self.last_frame_time = None

        self.predicting = False
        self.stopped = False
        self.ready = True

        self.model = None
    
    def load_model(self, model):
        self.model = model

        input_shape = model.layers[0].input_shape[0]
        input_shape = tuple(input_shape[1:])

        if input_shape != self.shape:
            print(f"Model input shape {input_shape} is not the same as the set shape {self.shape}. \nSetting the shape to fit the model.")
            self.shape = input_shape


    def start(self):
        if self.predicting:
            # Already started, so don't do anything
            return
        
        if self.model is None:
            print("No model loaded")
            return
        self.stopped = False
        threading.Thread(target=self.process, args=()).start()
    
    def stop(self):
        if not self.predicting:
            return
        self.stopped = True
        self.predicting = False

    def clear_prediction(self):
        lock.acquire()
        self.new_frame = False
        self.frame = None
        self.prediction = None
        self.average_prediction = None
        self.prediction_count = 0
        lock.release()

    def process(self):

        self.predicting = True

        while not self.stopped:
            lock.acquire()
            new_frame = self.new_frame
            frame = self.frame
            lock.release()

            if new_frame:
                self.new_frame = False
                self.predict_frame(frame)
                self.ready = True
            else:
                # If last new frame is more than the set prediction timeout, then clear current prediction
                diff = time.time() - self.last_frame_time
                if diff > self.prediction_timeout:
                    self.clear_prediction()
                time.sleep(1./1000)
    
    def put_frame(self, frame):
        if not self.predicting:
            return
        
        lock.acquire()
        self.frame = frame
        self.new_frame = True
        self.ready = False
        self.last_frame_time = time.time()

        lock.release()

    def predict_frame(self, frame):
        frame = cv2.resize(frame, self.shape)
        frame = np.expand_dims(frame, axis=0)

        self.prediction = self.model.predict(frame)
        self.prediction_count += 1
        if self.average_prediction is None:
            self.average_prediction = self.prediction
        else:
            self.average_prediction = [self.average_prediction[i] + (self.prediction[i] - self.average_prediction[i]) / self.prediction_count for i in range(len(self.average_prediction))]
