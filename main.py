import cv2
import json
import torch
import logging 
import argparse
import numpy as np
from PIL import Image
from pathlib import Path
from boxmot import DeepOCSORT
import datetime

class Watcher:
    def __init__(self, args):
        self.tracker = DeepOCSORT(model_weights=Path('checkpoints/osnet_x0_25_msmt17.pt'), device='cuda:0',fp16=True)
        self.device = args.device
        self.detector = torch.hub.load('ultralytics/yolov5', args.model, verbose=False).to(args.device)
        self.cap = cv2.VideoCapture(args.video)
        self.colors = np.random.randint(0, 255, size=(100, 3), dtype=np.uint8).tolist()

        self.formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
        self.logger = self.setup_logger('first_logger', 'first_logfile.log')
        self.logger.info('This is just info message')

    def setup_logger(self, name, log_file, level=logging.INFO):
        """To setup as many loggers as you want"""
    
        handler = logging.FileHandler(log_file)        
        handler.setFormatter(self.formatter)
    
        logger = logging.getLogger(name)
        logger.setLevel(level)
        logger.addHandler(handler)
    
        return logger
    
    def detect_person(self, img):
        results = self.detector(img).pandas().xyxy[0]
        boxes = [] #(x, y, x, y, conf, cls_id)
        for idx in results.index:
            cls_id = results['class'][idx]
            if cls_id == 0:
                x1 = results['xmin'][idx]
                y1 = results['ymin'][idx]
                x2 = results['xmax'][idx]
                y2 = results['ymax'][idx]
                conf = results['confidence'][idx]
                boxes.append([x1, y1, x2, y2, conf, cls_id])
        return np.array(boxes)
    
    def draw_boxes(self, frame, boxes):
        frame = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)
        for box in boxes:
            x1, y1, x2, y2, person_id, conf, cls_id = box
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), self.colors[int(person_id)], 2)
            cv2.putText(frame, f'id: {int(person_id)}', (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors[int(person_id)], 2)
        return frame
    
    def visualize(self, frame, boxes):
        frame = self.draw_boxes(frame, boxes)
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) == ord('q'):
            self.cap.release()

    def log_metadata(self, bboxes):
        output = []
        for box in bboxes:
            x1, y1, x2, y2, person_id, conf, cls_id = box
            center = (int((x1+x2)/2), int((y1+y2)/2))
            w = x2-x1
            h = y2-y1
            output.append([int(person_id), center, int(w), int(h)])
        self.logger.info(json.dumps(output))

    
    def run(self):
        tracked_boxes = []
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = Image.fromarray(frame)
            
            #.detect_person
            boxes = self.detect_person(frame)

            #.update tracker
            if boxes.shape[0] != 0:
                tracked_boxes = self.tracker.update(boxes, np.array(frame)) # --> (x, y, x, y, id, conf, cls_id)

            #.log_metadata
            self.log_metadata(tracked_boxes)

            #.Visualize
            self.visualize(frame, tracked_boxes)
            
def parse_args():
    parser = argparse.ArgumentParser(description='YOLOv5 detect')
    parser.add_argument('--video', type=str, default='/home/ashish/Videos/sp1.mp4', help='video path')
    parser.add_argument('--model', type=str, default='yolov5s', help='model name')
    parser.add_argument('--device', type=str, default='cuda', help='device')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    watcher = Watcher(args)
    watcher.run()




