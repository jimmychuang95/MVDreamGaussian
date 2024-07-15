# eval the clip-similarity for an input image and a geneated mesh
import cv2
import torch
import numpy as np
from torchvision import transforms as T
from transformers import CLIPModel, CLIPTokenizer, CLIPProcessor

# import kiui
# from kiui.render import GUI
from PIL import Image
import random
class CLIP:
    def __init__(self, device, model_name='openai/clip-vit-large-patch14'):

        self.device = device

        self.clip_model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
    
    def encode_image(self, image):
        # image: PIL, np.ndarray uint8 [H, W, 3]

        pixel_values = self.processor(images=image, return_tensors="pt").pixel_values.to(self.device)
        image_features = self.clip_model.get_image_features(pixel_values=pixel_values).to(self.device)

        image_features = image_features / image_features.norm(dim=-1,keepdim=True)  # normalize features

        return image_features

    def encode_text(self, text):
        # text: str

        inputs = self.processor(text=[text], padding=True, return_tensors="pt").to(self.device)
        text_features = self.clip_model.get_text_features(**inputs).to(self.device)

        text_features = text_features / text_features.norm(dim=-1,keepdim=True)  # normalize features

        return text_features


if __name__ == '__main__':
    import os
    import tqdm
    import argparse
    import json

    parser = argparse.ArgumentParser()

    parser.add_argument('--img_folder', type=str, default='/', help="eval img folder")
    
    opt = parser.parse_args()

    clip = CLIP('cuda', model_name='laion/CLIP-ViT-bigG-14-laion2B-39B-b160k')
    for num in range(0,10):
        folders = os.listdir(opt.img_folder)
        folders.sort()
        all_score = []
        results_json = {}
        for folder in folders:
            img_folder = os.path.join(opt.img_folder,folder,'save/it1200-test')
            img_list = [os.path.join(nm) for nm in os.listdir(img_folder) if nm[-3:] in ['jpg','png','gif']]
            img_list.sort(key=lambda x:int(x[:-4]))
            #
            results = []
            text = folder.split('@')[0]
            text = text.replace('_',' ')
            #
            text_features = clip.encode_text(text)



            img_list = random.sample(img_list,10)

            for img_name in img_list:
                path = os.path.join(img_folder,img_name)

                ref_img = Image.open(path)
                with torch.no_grad():
                    ref_features = clip.encode_image(ref_img)
                    similarity = (ref_features * text_features).sum(dim=-1).mean().item()
                    results.append(similarity)
            avg_similarity = np.mean(results)
            results_json[folder] = avg_similarity
            all_score.append(avg_similarity)

        all_mean = np.mean(all_score)
        results_json['all_mean'] = all_mean
        with open('GaussianDreamer.json','w',encoding='utf-8') as f:
            json.dump(results_json,f,indent = 4)

    


            
