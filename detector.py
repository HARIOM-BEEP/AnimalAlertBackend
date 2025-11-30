import numpy as np
from PIL import Image
import tflite_runtime.interpreter as tflite

class AnimalDetector:
    def __init__(self, model_path, label_path):
        self.interpreter = tflite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()

        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        with open(label_path, "r") as f:
            self.labels = [line.strip() for line in f.readlines()]

    def detect(self, image: Image.Image):
        img = image.resize((300, 300))
        img = np.expand_dims(img, axis=0)
        img = np.float32(img)

        self.interpreter.set_tensor(self.input_details[0]['index'], img)
        self.interpreter.invoke()

        boxes = self.interpreter.get_tensor(self.output_details[0]['index'])[0]
        class_ids = self.interpreter.get_tensor(self.output_details[1]['index'])[0]
        scores = self.interpreter.get_tensor(self.output_details[2]['index'])[0]

        # Find highest score
        max_idx = np.argmax(scores)
        confidence = float(scores[max_idx])

        if confidence < 0.55:
            return None, 0.0

        label = self.labels[int(class_ids[max_idx])]
        return label, confidence
