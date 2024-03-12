import torch
from PIL import Image
from transformers import OwlViTProcessor, OwlViTForObjectDetection
# from transformers import Owlv2Processor, Owlv2ForObjectDetection
import numpy as np
import cv2
from vision.nms import nms
class owl_vit():
    def __init__(self):
        print("start loading owl-vit")
        self.processor_model = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
        self.model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32").cuda()
        self.model.eval()
        print("End Loading")

    def query_text(self, texts):
        texts = [["a photo of a {}".format(text) for text in texts]]
        self.text = texts

    def detect(self, frame, thres = 0.25, vis=False):
        # image_uint8 = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_uint8 = frame
        # image = image_uint8.astype(np.float32) / 255.0
        h_o, w_o, _ = image_uint8.shape
        image = Image.fromarray(image_uint8)
        inputs = self.processor_model(text=self.text, images=image, return_tensors="pt")
        inputs = {name: tensor.cuda() for name, tensor in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
        # outputs = [{k: v.cpu() for k, v in outputs.items()}]
        # Target image sizes (height, width) to rescale box predictions [batch_size, 2]
        # target_sizes = torch.Tensor([image.size[::-1]])
        target_sizes = torch.Tensor([[h_o, w_o]]).cuda()
        # Convert outputs (bounding boxes and class logits) to COCO API
        results = self.processor_model.post_process_object_detection(outputs=outputs, 
                                target_sizes=target_sizes, threshold=thres)
        i = 0  # Retrieve predictions for the first image for the corresponding text queries
        text = self.text[i]
        boxes, scores, labels = results[i]["boxes"], results[i]["scores"], results[i]["labels"]
        print(scores)
        # NMS
        nms_threshold = 0.1
        nmsed_indices = nms(
                boxes.cpu().numpy(),
                scores.cpu().numpy(),
                thresh=nms_threshold
                )
        boxes = boxes[nmsed_indices].cpu().numpy()
        scores = scores[nmsed_indices].cpu().numpy()
        labels = labels[nmsed_indices].cpu().numpy()

        cv_image = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        boxes_res = []
        new_label = []
        mask = np.zeros((h_o, w_o), dtype=np.uint8)
        for i, (box, score, label) in enumerate(zip(boxes, scores, labels)):
            # if score < thres: continue
            box = [round(i, 2) for i in box.tolist()]
            y1, x1, y2, x2 = box
            x1= int(x1); y1= int(y1); x2= int(x2); y2= int(y2)
            cv2.rectangle(cv_image,(y1,x1),(y2,x2),(255,0,0),3)
            cv2.putText(cv_image, text[label], (y1, x1), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            print(f"Detected {text[label]} with confidence {round(score.item(), 3)} at location {box}")
            mask[x1:x2, y1:y2] = i+1
            boxes_res.append([y1,x1,y2,x2])
            new_label.append(text[label])
        return cv_image,mask, boxes_res, new_label