import cv2
import numpy as np
import matplotlib.pyplot as plt

class MaskRCNN:
    """MaskRCNN class take three parameters model weights and config file  and lable_file path"""
    def __init__(self,model,config,lable_file):
        self.model=model
        self.config=config
        self.net = cv2.dnn.readNetFromTensorflow(self.model,self.config)
        self.lable_file=lable_file

    """feed the input towards network""" 
    def feed_network(self,img):
        blob = cv2.dnn.blobFromImage(img, swapRB=True)
        self.net.setInput(blob)
        boxes, masks = self.net.forward(["detection_out_final", "detection_masks"])
        return boxes, masks

    """read labels from file""" 
    def read_label(self):
        # label name
        lines = []
        with open(self.lable_file) as f:
            for line in f:
                lines.append(line.rstrip())
        return lines

    """detection and draw bounding boxes"""  
    def detection(self,img,detection_count):
        width=img.shape[1]
        height=img.shape[0]
        for i in range(detection_count):
            box = boxes[0, 0, i]
            class_id = box[1]
            score = box[2]
            if score > 0.7:
                # Boxcoordinates
                x = int(box[3] * width)
                y = int(box[4] * height)
                x2 = int(box[5] * width)
                y2 = int(box[6] * height)
                cv2.rectangle(img, (x, y), (x2, y2), (255, 0, 0), 3)
                label=self.read_label()
                #print(label)
                label = "{}: {:.4f}".format(label[int(class_id)], score)
                cv2.putText(img, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        return img

if __name__ == "__main__":
   
    # Loading Mask RCNN
    model="dnn/frozen_inference_graph_coco.pb"
    config="dnn/mask_rcnn_inception_v2_coco_2018_01_28.pbtxt"
    lable_file='dnn/label.txt'
    opencv_dnn= MaskRCNN(model,config,lable_file)
    cap=cv2.VideoCapture(0)

    while (cap.isOpened):
        ret, img=cap.read()
        boxes, masks=opencv_dnn.feed_network(img)
        detection_count = boxes.shape[2]
        img=opencv_dnn.detection(img,detection_count)
        cv2.imshow("Image", img)    
        if cv2.waitKey(27) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


    

