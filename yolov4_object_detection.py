#----import lib----
import cv2
import numpy as np
import matplotlib.pyplot as plt
import time

class YoloObjectDetection(object):
    """docstring for YoloObject detection"""
    def __init__(self,model,weights,lable_file):
        self.model=model
        self.weights=weights
        self.net = cv2.dnn.readNetFromDarknet(self.model,self.weights)
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
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

        blob = cv2.dnn.blobFromImage(img, 1/255.0, (416, 416),(0,0,0),swapRB=True, crop=False)
        # determine the output layer
        output_layers=self.net.getUnconnectedOutLayers()
        layers = self.net.getLayerNames()
        layers = [layers[i - 1] for i in output_layers]
        self.net.setInput(blob)
        outputs = self.net.forward(layers)
            
        return outputs

    """detection and draw bounding boxes""" 
    def detection(self,img,outputs):
        width=img.shape[1]
        height=img.shape[0]
        boxes=[]
        box_confidences=[]
        class_ids=[]
        label_name=self.read_label()
        colors = np.random.randint(0, 255, size=(len(label_name), 3), dtype='uint8')   
        for output in outputs:
            for out in output:
                scores=out[5:]
                class_id=np.argmax(scores)
                confidence=scores[class_id]
                
                if confidence>0.5:
                    width_ = int(out[2] * width)
                    height_ = int(out[4] * height)
                    x = int((out[0]*width)-width_/2)
                    y = int((out[1]*height)-height_/2)
                    boxes.append([x,y,width_,height_])
                    box_confidences.append(float(confidence))
                    class_ids.append(class_id)
                    
        indices = cv2.dnn.NMSBoxes(boxes, box_confidences, 0.5, 0.4)   

        if len(indices) > 0:
            
            for i in indices.flatten():
                (x, y) = (boxes[i][0], boxes[i][1])
                (w, h) = (boxes[i][2], boxes[i][3])
                color = [int(c) for c in colors[class_ids[i]]]

                cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
                label = "{}: {:.4f}".format(label_name[class_ids[i]], box_confidences[i])
                cv2.putText(img, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                
        return img

if __name__ == "__main__":

    # Loading YOLO v4 model confi and weights
    model="dnn/yolov4/yolov4.cfg"
    weights="dnn/yolov4/yolov4.weights"
    lable_file='dnn/label_yolo.txt'

    opencv_dnn= YoloObjectDetection(model,weights,lable_file)
    cap=cv2.VideoCapture(0)
    frame_rate = 10
    prev = 0
    
    while (cap.isOpened):
        time_elapsed = time.time() - prev
        ret, img=cap.read()
        if time_elapsed > 1.0/frame_rate:
            prev = time.time()
            boxes=opencv_dnn.feed_network(img)
            img=opencv_dnn.detection(img,boxes)
            cv2.imshow("Image", img)    
            if cv2.waitKey(27) & 0xFF == ord('q'):
                 break

    cap.release()
    cv2.destroyAllWindows()
