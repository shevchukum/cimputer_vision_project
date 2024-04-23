import csv
import cv2
from pytube import YouTube
from ultralytics import YOLO
import json
import torch
from tabulate import tabulate
import time
import torchvision.transforms as transforms
import torchvision.models.detection as detection

def main():
    # List of 79 classes used for YOLO training
    yolo_names = {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 
                  5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light', 
                  10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 
                  13: 'bench', 14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 
                  19: 'cow', 20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe', 
                  24: 'backpack', 25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase', 
                  29: 'frisbee', 30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite', 
                  34: 'baseball bat', 35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 
                  38: 'tennis racket', 39: 'bottle', 40: 'wine glass', 41: 'cup', 42: 'fork', 
                  43: 'knife', 44: 'spoon', 45: 'bowl', 46: 'banana', 47: 'apple', 48: 'sandwich', 
                  49: 'orange', 50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza', 
                  54: 'donut', 55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant', 
                  59: 'bed', 60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop', 
                  64: 'mouse', 65: 'remote', 66: 'keyboard', 67: 'cell phone', 68: 'microwave', 
                  69: 'oven', 70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book', 
                  74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier', 
                  79: 'toothbrush'}

    # List of 107 classes in COCO dataset used for training RetinaNet and Faster R-CNN
    coco_names = [
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", 
        "traffic light", "fire hydrant", "steet sign", "stop sign", "parking meter", "bench", 
        "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", 
        "hat", "backpack", "umbrella", "shoe", "eyeglasses", "handbag", "tie", "suitcase", "frisbee",
        "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", 
        "surfboard", "tennis racket", "bottle", "plate", "wine glass", "cup", "fork", "knife", 
        "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog",
        "pizza", "donut", "cake", "chair", "sofa", "potted plant", "bed", "mirror", "dining table",
        "window", "desk", "toilet", "door", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
        "microwave", "oven", "toaster", "sink", "refrigerator", "blender", "book", "clock", "vase",
        "scissors", "teddy bear", "hair drier", "toothbrush", "hairbrush"]
    
    # Define result variables
    obs_num = 0                                              # Number of valid observations
    res_yolo = res_retina = res_faster = [0, 0, 0, 0, 0]     # Result vectors for each model
    samples = 650                                            # Number of frames to check in Youtube database

    # Load models
    model_yolo = YOLO('yolov9c.pt')
    
    retinanet_model = detection.retinanet_resnet50_fpn(pretrained=True)
    retinanet_model.eval()
    
    faster_rcnn_model = detection.fasterrcnn_resnet50_fpn(weights=detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
    faster_rcnn_model.eval()
    
    # Open the CSV file with Yutube database
    with open(r'C:\users\shevc\Computer_Vision\database.csv', 'r') as csvfile:
        for t in range(samples):

            # Read a line and split
            line = csvfile.readline()
            elements_list = line.split(',')
            
            # If frame true class is not in YOLO class list - skip the line   
            true_class = elements_list[3]
            if check_frame(true_class) == False: continue  

            # Define Youtube ID, frame start time and box coordinates 
            youtube_id = elements_list[0]
            start_time = int(int(elements_list[1]) / 1000)
            xmin = float(elements_list[6])       # Left-most location of bounding box (relative to frame size)
            xmax = float(elements_list[7])       # Right-most location of bounding box (relative to frame size)
            ymin = float(elements_list[8])       # Top-most location of bounding box (relative to frame size)
            ymax = float(elements_list[9])       # Bottom-most location of bounding box (relative to frame size)

            # Creating the true obj box = xmin, ymin, xmax, ymax. 
            # If objest is absent -> xmin = -1.0
            true_box = [xmin, ymin, xmax, ymax]
            
            # Reading the frame and saving it to temp jpg file        
            if not pre_process(youtube_id, start_time): continue
            
            # Now observation is valid -> counter + 1
            obs_num += 1
            
            # Getting outputs from models
            result_yolo = yolo_inference(true_class, true_box, model_yolo)
            result_retina = RetinaNet_inference(true_class, true_box, retinanet_model)
            result_faster = Faster_inference(true_class, true_box, faster_rcnn_model)
            
            # For each model adding new output with the sum of the previous ones
            # This allows to calculate all metrics after each obeservation and control the process
            res_yolo = [x + y for x, y in zip(res_yolo, result_yolo)]
            res_retina = [x + y for x, y in zip(res_retina, result_retina)]
            res_faster = [x + y for x, y in zip(res_faster, result_faster)]
        
            # Calculating metrics for each model
            average_IuO_y = res_yolo[3] / max(1, res_yolo[0])
            precision_y = res_yolo[0]/max(1, (res_yolo[0] + res_yolo[1]))
            recall_y = res_yolo[0]/max(1, (res_yolo[0] + res_yolo[2]))
            F1_y = 2 * precision_y * recall_y / max(1, (precision_y + recall_y))
            accuracy_y = (obs_num - res_yolo[2] - res_yolo[1]) / obs_num
            elapsed_time_y = res_yolo[4]

            average_IuO_r = res_retina[3] / max(1, res_retina[0])
            precision_r = res_retina[0]/max(1, (res_retina[0] + res_retina[1]))
            recall_r = res_retina[0]/max(1, (res_retina[0] + res_retina[2]))
            F1_r = 2 * precision_r * recall_r / max(1, (precision_r + recall_r))
            accuracy_r = (obs_num - res_retina[2] - res_retina[1]) / obs_num
            elapsed_time_r = res_retina[4]

            average_IuO_f = res_faster[3] / max(1, res_faster[0])
            precision_f = res_faster[0]/max(1, (res_faster[0] + res_faster[1]))
            recall_f = res_faster[0]/max(1, (res_faster[0] + res_faster[2]))
            F1_f = 2 * precision_f * recall_f / max(1, (precision_f + recall_f))
            accuracy_f = (obs_num - res_faster[2] - res_faster[1]) / obs_num
            elapsed_time_f = res_faster[4]

            # Printing metrics table
            data = [["YOLOv9-c", precision_y, recall_y, F1_y, average_IuO_y, accuracy_y, elapsed_time_y],
                    ["RetinaNet", precision_r, recall_r, F1_r, average_IuO_r, accuracy_r, elapsed_time_r],
                    ["Faster R-CNN", precision_f, recall_f, F1_f, average_IuO_f, accuracy_f, elapsed_time_f]
                   ]
            headers = ["Model", "Precision", "Recall", "F1", "a_IuO", "Accuracy", "Elapsed Time"]
            table = tabulate(data, headers=headers, tablefmt="grid")
            print(table)
        
def check_frame(name):
    ''' Checking if the frame label is in yolo_name (if it is then it is also in coco).
    '''
    for elem in class_names.values():
        if elem == name: return True
    return False
    
def pre_process(youtube_id, frame_time):
    ''' Function reading the frame from Youtube video, resize it to 640x640 and saving to jpg file.
    '''
    # Retrieve the YouTube video stream
    url = f'https://www.youtube.com/watch?v={youtube_id}'
    youtube = YouTube(url)
    try: 
        stream = youtube.streams.get_highest_resolution()
    except Exception as e:
        print("Exception: ", e)               # Some videos are deleted from Youtube or have access restriction
        return False

    # Open the video stream and get frame rate
    cap = cv2.VideoCapture(stream.url)
    fps = cap.get(cv2.CAP_PROP_FPS)
 
    # Calculate the frame number and seak the target frame
    target_frame = int(frame_time * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)

    # Read frame from the video stream
    ret, frame = cap.read()
    if not ret: return False

    # Resize frame to 640x640 and convert it to RGB
    frame = cv2.resize(frame, (640, 640))
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Save frame_rgb to a temporal file
    with open('temp_image.jpg', 'wb') as f:
        f.write(cv2.imencode('.jpg', frame_rgb)[1])
    return True

def yolo_inference(true_class, true_box, model):
    ''' Function making inference with YOLO on 'temp_image.jpg' and
        returning a list [TP, FP, FN, iou, elapsed_time] for benchmark calculations.
    '''

    # Run YOLO model and measure elapsed_time
    start = time.time()
    results = model('temp_image.jpg')
    elapsed_time = time.time() - start
    
    # Producing TP, TN, FP, FN
    json_res = results[0].tojson()
    dict_res = json.loads(json_res)
    if len(dict_res) == 0 and true_box[0] == -1: 
        return [0, 0, 0, 0, elapsed_time]     # Absent object, no detection = TN
    elif len(dict_res) == 0 and true_box[0] != -1: 
        return [0, 1, 0, 0, elapsed_time]     # Present obj, no detection = FP
    elif len(dict_res) != 0 and true_box[0] == -1:
        for elem in dict_res:
            if elem["name"] == true_class:
                return [0, 0, 1, 0, elapsed_time]     # Absent obj, obj detection = FN
        return [0, 0, 0, 0, elapsed_time]     # Absent obj, no obj detection = TN
        
    # For TP producing IoU
    max_iou = 0
    result = [0, 1, 0, 0, elapsed_time]        # By default FP
    for elem in dict_res:
        if elem["name"] == true_class:
            box_list = list(elem["box"].values())
            box_list = [value / 640 for value in box_list]   # Normalizing
            c_iou = calculate_iou(box_list, true_box) 
            if c_iou == 0: continue
            max_iou = max(max_iou, c_iou)
            result = [1, 0, 0, max_iou, elapsed_time]  # TP with the best matching box IoU
    return result

def RetinaNet_inference(true_class, true_box, model):
    ''' Function making inference with Retina on 'temp_image.jpg' and 
        returning a list [TP, FP, FN, iou, elapsed_time] for benchmark calculations.
    '''
        
    # Starting timer and loading the image using OpenCV
    start = time.time()
    image = cv2.imread('temp_image.jpg')

    # Converting image to normalized tensor and add batch dimension
    image_normalized = image.astype(np.float32) / 255.0
    input_tensor = torch.tensor(image_normalized).permute(2, 0, 1).unsqueeze(0)
    
    # Perform inference with RetinaNet and fixing the elapsed time
    with torch.no_grad(): output = model(input_tensor)
    elapsed_time = time.time() - start

    # Extract bounding boxes and class labels from the output and cutting them
    boxes = output[0]['boxes'].cpu().numpy()  # Bounding boxes (xmin, ymin, xmax, ymax)
    labels = output[0]['labels'].cpu().numpy()  # Class labels
    boxes = boxes[:10]
    labels = labels[:10]
    
    # Producing TP, TN, FP, FN    
    if len(labels) == 0 and true_box[0] == -1: 
        return [0, 0, 0, 0, elapsed_time]     # Absent object, no detection = TN
    elif len(labels) == 0 and true_box[0] != -1: 
        return [0, 1, 0, 0, elapsed_time]     # Present obj, no detection = FP
    elif len(labels) != 0 and true_box[0] == -1:
        for elem in labels:
            if coco_names[elem-1] == true_class:
                return [0, 0, 1, 0, elapsed_time]     # Absent obj, obj detection = FN
        return [0, 0, 0, 0, elapsed_time]     # Absent obj, no obj detection = TN
        
    # For TP producing IoU
    max_iou = 0
    result = [0, 1, 0, 0, elapsed_time]        # By default FP
    for x in range(len(labels)):
        if coco_names[labels[x]-1] == true_class:
            box_list = [value / 640 for value in boxes[x]]   # Normalizing
            c_iou = calculate_iou(box_list, true_box) 
            if c_iou == 0: continue
            max_iou = max(max_iou, c_iou)
            result = [1, 0, 0, max_iou, elapsed_time]  # TP with the best matching box IoU
    return result

def Faster_inference(true_class, true_box, model):
    ''' Function making inference with Faster R-CNN on 'temp_image.jpg' and 
        returning a list [TP, FP, FN, iou, elapsed_time] for benchmark calculations.
    '''

    # Starting timer and loading the image using OpenCV
    start = time.time()
    image = cv2.imread('temp_image.jpg')

    # Converting image to normalized tensor and add batch dimension
    image_normalized = image.astype(np.float32) / 255.0
    input_tensor = torch.tensor(image_normalized).permute(2, 0, 1).unsqueeze(0)
    
    # Perform inference with the Faster R-CNN model and fix timer
    with torch.no_grad(): predictions = faster_rcnn_model(input_tensor)
    elapsed_time = time.time() - start

    # Extract bounding boxes and class labels and cutting them
    boxes = predictions[0]['boxes'].cpu().numpy()
    labels = predictions[0]['labels'].cpu().numpy()
    boxes = boxes[:10]
    labels = labels[:10]

    # Producing TP, TN, FP, FN 
    if len(labels) == 0 and true_box[0] == -1: 
        return [0, 0, 0, 0, elapsed_time]     # Absent object, no detection = TN
    elif len(labels) == 0 and true_box[0] != -1: 
        return [0, 1, 0, 0, elapsed_time]     # Present obj, no detection = FP
    elif len(labels) != 0 and true_box[0] == -1:
        for elem in labels:
            if coco_names[-1] == true_class:
                return [0, 0, 1, 0, elapsed_time]     # Absent obj, obj detection = FN
        return [0, 0, 0, 0, elapsed_time]     # Absent obj, no obj detection = TN
        
    # For TP producing IoU
    max_iou = 0
    result = [0, 1, 0, 0, elapsed_time]        # By default FP
    for x in range(len(labels)):
        if coco_names[labels[x]-1] == true_class:
            box_list = [value / 640 for value in boxes[x]]   # Normalizing
            c_iou = calculate_iou(box_list, true_box) 
            if c_iou == 0: continue
            max_iou = max(max_iou, c_iou)
            result = [1, 0, 0, max_iou, elapsed_time]  # TP with the best matching box IoU
    return result
        
def calculate_iou(box1, box2):
    """
    Calculate Intersection over Union (IoU) between two bounding boxes.
    box1 and box2 should be in format (xmin, ymin, xmax, ymax), both in 
    normalized values.
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area_box1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area_box2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area_box1 + area_box2 - intersection

    if union == 0:
        return 0                                # If the union is zero, IoU is undefined
    else:
        return intersection / union


            
if __name__ == "__main__":
    main()