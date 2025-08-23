# BLIP captioning example: how to use BLIP to caption a single image

import requests
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import numpy as np
import torch
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")

### statue scene
# folder_path_each[i] = "032dee9fb0a8bc1b90871dc5fe950080d0bcd3caf166447f44e60ca50ac04ec7"
### market scene
# folder_path_each[i] = "25231e5e062b71d1f9b0463219e63a2383d55f3b2cec95f50e20f044d60ef4f6"
### restaurant scene
# folder_path_each[i] = "c37726ce770ac50a2cf5c0f43022f0268e26da0d777cd8e3a3418c4eed03fd94"
### inside a restaurant
# 07d9f9724ca854fae07cb4c57d7ea22bf667d5decd4058f547728922f909956bs

image_path = "/your/path/to/DL3DV-10K/c37726ce770ac50a2cf5c0f43022f0268e26da0d777cd8e3a3418c4eed03fd94/gaussian_splat/images_4/frame_00001.png"
raw_image = Image.open(image_path).convert('RGB')
raw_image = torch.tensor(np.array(raw_image)).permute(2,0,1)
# unconditional image captioning
inputs = processor(raw_image, return_tensors="pt")

out = model.generate(**inputs)
print(processor.decode(out[0], skip_special_tokens=True))
