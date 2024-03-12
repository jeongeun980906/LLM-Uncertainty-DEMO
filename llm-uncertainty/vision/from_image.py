import cv2
import numpy as np
from openai import OpenAI
# from PIL import Image
import requests
import json, io
import base64
from ultralytics import YOLO

import matplotlib.pyplot as plt
import numpy as np

colors = [plt.cm.Dark2(i) for i in range(8)]
colors = [(int(255*color[0]),int(255*color[1]),int(255*color[2])) for color in colors]
# print(colors[1])
class gpt4_v_helper():
    def __init__(self, key_file_path='./key/key.txt', max_tokens = 300, temperature = 0.0, n = 1, stop = [], VERBOSE=True):
        api_key = self.set_openai_api_key_from_txt(key_file_path,VERBOSE=VERBOSE)
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
            }
        self.client = OpenAI(api_key=api_key)
        self.messages = []
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.n = n
        self.stop = stop
        self.model = YOLO('yolov8n.pt')

    def set_system_prompt(self):
        system_prompt = 'You are an agent who describe be what you can see in the picture'
        self.messages.append({"role": "system", "content": system_prompt})

    def SoM_style(self, img_path, boxes, clss, confs, human_only=False):
        img = cv2.imread(img_path)
        boxes = boxes.cpu().numpy()
        # print(img.shape, boxes)
        i = 0
        if human_only:
            new_indx = np.argsort(boxes[:,0]) #[::-1]
        else: 
            new_indx = np.argsort(boxes[:,0])
        # print(new_indx)
        boxes = boxes[new_indx]
        clss = clss.cpu().numpy()[new_indx]
        confs = confs.cpu().numpy()[new_indx]
        res_box = []
        mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
        for box, cls, conf in zip(boxes, clss,confs):
            if conf < 0.5 and human_only: continue
            if not human_only and cls == 60: continue
            if human_only and cls != 0: continue
            box = [int(i)for i in box]
            if human_only: 
                # box[1] = box[0]+int((box[2]-box[0])*0.4)
                box[3] = box[1]+int((box[3]-box[1])*0.8)
            img = cv2.rectangle(img,[box[0],box[1]],[box[2],box[3]],
                                color=(50,50,50), thickness=2) # colors[i]
            
            img = cv2.rectangle(img,[box[0]-5,box[1]-20],[box[0]+20,box[1]+20],
                                color=(0,0,0), thickness=-1)
            img = cv2.putText(img,str(i),[box[0]-5,box[1]+5], cv2.FONT_HERSHEY_DUPLEX,
                              color=(255,255,255),fontScale=1.0)
            i += 1
            mask[box[1]:box[3], box[0]:box[2]] = i
            res_box.append(box)
        img_path_split = img_path.split(".jpg")
        img_path_split[0] += '_det'
        new_name = img_path_split[0]+'.jpg'
        cv2.imwrite(new_name, img)
        return new_name, i, res_box, mask

    def detection(self,img_path, cands, human=False):
        results = self.model(img_path) 
        # print(results, results[0].boxes.cls)
        boxes = results[0].boxes.xyxy  # print the Boxes object containing the detection bounding boxes
        cls = results[0].boxes.cls
        conf = results[0].boxes.conf
        new_path, len_box, res_box, mask = self.SoM_style(img_path,boxes, cls, conf, human)
        prompt = 'List the objects available in the scene. The following is the class of the objects: '
        prompt += str(cands)
        prompt += "please do not mention any other objects outside the mentioned class"
        prompt += "If you see multiple objects in the same class, "
        prompt += 'or if you see a people wearing same color of the shirt,'
        prompt += 'please follow the notation as follows: left_obj or right_obj'
        prompt+= 'such as, left_pepsi_can, right_pepsi_can, left_person_wearing_red_shirt, right_person_wearing_red_shirt'
        prompt +='do not forget to add left, right tag to all of the objects that has the same class name'
        prompt += 'please follow this format for your output the front number is the mark: [0: obj_1, 1: obj_2]'
        if human: prompt += 'Ignore the color of a robot, blue. Do not be confused with a robot color and the shirt color, where the robot is blue. And please predict the person inside each bounding boxes. Do not confuse with background color white'
        while True:
            res = self.ask(prompt,[new_path],True)
            obj_list = res.split(",")
            if len(obj_list) >= len_box: break
            else: prompt += "Nope. You have not labeled all of the objects or humans. You missed number {}".format(len_box-1)
        refined_list = []
        for obj in obj_list:
            obj = obj.split(":")[1].strip()
            obj = obj.replace("]","")
            refined_list.append(obj)
        return refined_list, res_box, mask

    def ask(self, question, image_paths = [], APPEND=True):
        content = [{"type": "text", "text": question}]
        for image_path in image_paths:
            base64_image = self.encode_image(image_path)
            image_message = {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}"
                    }
                }
            content.append(image_message)
        user_message = {
            "role": "user",
            "content": content
        }
        payload = self.create_payload()
        self.messages.append(user_message)
        response = requests.post("https://api.openai.com/v1/chat/completions", headers=self.headers, json=payload)
        res = response.json()['choices'][0]['message']['content']
        print(response.json()['choices'][0]['message'])
        if APPEND:
            self.messages.append({"role": "assistant", "content": res})
        return res
    
    def reset(self):
        self.messages = []
        
    def set_openai_api_key_from_txt(self, key_path='./key/rilab_key.txt',VERBOSE=True):
        """
            Set OpenAI API Key from a txt file
        """
        with open(key_path, 'r') as f: 
            OPENAI_API_KEY = f.read()
        if VERBOSE:
            print("OPENAI KEY SET")
        return OPENAI_API_KEY
    
    def create_payload(self):
        payload = {
            "model": "gpt-4-vision-preview",
            "messages": self.messages,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "n": self.n
        }
        if len(self.stop) > 0:
            payload["stop"] = self.stop
        return payload
    
    # Function to encode the image
    def encode_image(self, image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
# class detection():
#     def __init__(self,queries, person=False):
#         self.model = owl_vit()
#         self.object_queries = queries
#         self.model.query_text(queries)
#         print("finish loading model")

#     def detect_human(self, human_queries, save_indx = None):
#         self.model.query_text(human_queries)
#         human_cv = cv2.imread("./data/det/human_ori_{}.jpg".format(save_indx))
#         # inp = cv2.cvtColor(self.human_cv, cv2.COLOR_BGR2RGB)
#         im_color,_, boxes, found_object_names = self.model.detect(human_cv, thres=0.1)
#         if save_indx is not None:
#             im_color = cv2.cvtColor(im_color, cv2.COLOR_BGR2RGB)
#             cv2.imwrite("./data/det/human_det_offline_{}.jpg".format(save_indx), im_color)
#         return im_color, found_object_names
    
#     def detect_object(self, save_indx = None):
#         self.model.query_text(self.object_queries)
#         print("start detecting")
#         inp = cv2.imread("./data/det/ori_{}.jpg".format(save_indx)) #cv2.cvtColor(self.rgb_cv, cv2.COLOR_BGR2RGB)
#         im_color,index_mask, boxes, found_object_names = self.model.detect(inp, thres=0.25)
#         if save_indx is not None:
#             # im_color = cv2.cvtColor(im_color, cv2.COLOR_BGR2RGB)
#             cv2.imwrite("./data/det/det_offline_{}.jpg".format(save_indx), im_color)
#         return im_color, found_object_names