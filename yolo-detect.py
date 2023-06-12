import numpy as np
import cv2
import jsearch
import os
import time

config = "config_files/config.json"
raw_folder = "../flask-server/received"
processed_folder = "processed"
file_timeout = 5

def format_yolov5(frame):
    row, col, _ = frame.shape
    _max = max(col, row)
    result = np.zeros((_max, _max, 3), np.uint8)
    result[0:row, 0:col] = frame
    return result

def get_oldest_file(folder_path):
    files = os.listdir(folder_path)
    if not files:
        return None  

    files.sort()  
    image_name = files[0]
    folder_path = os.path.normpath(folder_path)
    oldest_file_path = os.path.join(folder_path, image_name)
    return image_name,oldest_file_path

def extract_nodeID(filename):
    nodeID = filename[:6]
    timeStamp = filename[7:26]
    return nodeID, timeStamp

def log_area(timeStamp, nodeID, area_occupied):
    line = f"{str(timeStamp)} : {str(nodeID)} : {str(area_occupied)}\n"
    with open("output.txt", "a") as file:
        file.write(line)

while True:
    while True:
        image_name,image_path = get_oldest_file(raw_folder)
        if image_path is not None:
            print("Processing image found:", image_path)
            break  # Exit the while loop if a file is found
        
        # If no file is found, continue the loop
        print("No file found. Waiting...")
        time.sleep(file_timeout * 60)
        file_timeout = min(file_timeout + 5, 30)

    nodeID,timeStamp = extract_nodeID(image_name)

    image = cv2.imread(image_path)
    destination_file = os.path.join(processed_folder, os.path.basename(image_path))
    os.rename(image_path, destination_file)

################################# OUTPUT ARRAY PROCESSING ################################################
    input_image = format_yolov5(image)                 # Input image into net
    blob = cv2.dnn.blobFromImage(input_image , 1/255.0, (640, 640), swapRB=True)

    net = cv2.dnn.readNet('config_files/best (3).onnx')
    net.setInput(blob)
    predictions = net.forward()

    class_ids = []
    confidences = []
    boxes = []

    output_data = predictions[0]

    image_width, image_height, _ = input_image.shape
    x_factor = image_width / 640
    y_factor =  image_height / 640

    for r in range(25200):
        row = output_data[r]
        confidence = row[4]
        if confidence >= 0.4:

            classes_scores = row[5:]
            _, _, _, max_indx = cv2.minMaxLoc(classes_scores)
            class_id = max_indx[1]
            if (classes_scores[class_id] > .25):

                confidences.append(confidence)

                class_ids.append(class_id)

                x, y, w, h = row[0].item(), row[1].item(), row[2].item(), row[3].item() 
                left = int((x - 0.5 * w) * x_factor)
                top = int((y - 0.5 * h) * y_factor)
                width = int(w * x_factor)
                height = int(h * y_factor)
                box = np.array([left, top, width, height])
                boxes.append(box)

    class_list = []
    with open("config_files/classes.txt", "r") as f:
        class_list = [cname.strip() for cname in f.readlines()]

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.25, 0.45) 

    result_class_ids = []
    result_confidences = []
    result_boxes = []

    for i in indexes:
        result_confidences.append(confidences[i])
        result_class_ids.append(class_ids[i])
        result_boxes.append(boxes[i])

    for i in range(len(result_class_ids)):

        box = result_boxes[i]
        class_id = result_class_ids[i]

        cv2.rectangle(image, box, (0, 255, 255), 2)
        cv2.rectangle(image, (box[0], box[1] - 20), (box[0] + box[2], box[1]), (0, 255, 255), -1)
        cv2.putText(image, class_list[class_id], (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, .5, (0,0,0))

    cv2.imwrite("misc/detection.png", image)
################################# OUTPUT ARRAY PROCESSING ################################################


    area_occupied = 0
    total_area = int(image_width * image_height)
    
    for i in range(len(result_boxes)):
        box = result_boxes[i]
        area = int(box[2] * box[3]) 
        print("independant area occupied:",area)
        area_occupied = int(area_occupied + area)
    if area_occupied is not None :  
        area_occupied = round(100*(area_occupied / total_area),1)
        log_area(timeStamp,nodeID,area_occupied)            

