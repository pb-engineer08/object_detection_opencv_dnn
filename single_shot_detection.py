import cv2
import numpy as np
import matplotlib.pyplot as plt
import time

class MobileNetSSD:
    """MobileNetSSD class take three parameters model weights and config file  and lable_file path"""
    def __init__(self,model,config,lable_file):
        self.model=model
        self.config=config
        self.net = cv2.dnn.readNetFromCaffe(self.model,self.config)
        self.lable_file=lable_file

    """read labels from file""" 
    def read_label(self):
        # label name
        lines = []
        with open(lable_file) as f:
            for line in f:
                lines.append(line.rstrip())
        return lines

    """feed the input towards network""" 
    def feed_network(self,img):
        blob = cv2.dnn.blobFromImage(img, 0.007843, (300, 300), 127.5)
        #img=cv2.resize(img,(300,300))
        #img=np.expand_dims(img, axis=0)
        #img=np.rollaxis(img, 3, 1) 
        self.net.setInput(blob)
        boxes = self.net.forward()

        return boxes

    """detection and draw bounding boxes"""  
    def detection(self,img,detection_count,boxes):
        width=img.shape[1]
        height=img.shape[0]
        label_name=self.read_label()
        for i in range(detection_count):
            box = boxes[0, 0, i]
            class_id = box[1]
            score = box[2]
            if score > 0.7:
                # Get box Coordinates
                x = int(box[3] * width)
                y = int(box[4] * height)
                x2 = int(box[5] * width)
                y2 = int(box[6] * height)
                roi = img[y: y2, x: x2]
                cv2.rectangle(img, (x, y), (x2, y2), (255, 0, 0), 3)
                label = "{}: {:.2f}%".format(label_name[int(class_id)],score*100)
                cv2.putText(img, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(255, 0, 0), 2)
    
        return img

if __name__ == "__main__":
    # Loading MobileNetSSD model

    model="dnn/MobileNetSSD_deploy.prototxt"
    config="dnn/MobileNetSSD_deploy.caffemodel"
    lable_file='dnn/label_mblssd.txt'
    opencv_dnn= MobileNetSSD(model,config,lable_file)

    cap=cv2.VideoCapture(0)
    frame_rate = 10
    prev = 0
    while (cap.isOpened):
        time_elapsed = time.time() - prev
        ret, img=cap.read()
        if time_elapsed > 1.0/frame_rate:
            prev = time.time()
            boxes=opencv_dnn.feed_network(img)
            #print(boxes[0, 0, 1])
            detection_count = boxes.shape[2]

            img=opencv_dnn.detection(img,detection_count,boxes)
        cv2.imshow("Image", img)    
        if cv2.waitKey(27) & 0xFF == ord('q'):
            break
        #This breaks on 'q' key

    cap.release()
    cv2.destroyAllWindows()
