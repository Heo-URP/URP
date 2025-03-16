import torch
import numpy as np
import cv2
from sam2.sam2.build_sam import build_sam2
from sam2.sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from PIL import Image, ImageDraw

class SAM:
    def __init__(self, sam_checkpoint, model_cfg, device='mps'):
        self.device = device
        
        sam2 = build_sam2(model_cfg, sam_checkpoint, device=device)
        self.mask_generator = SAM2AutomaticMaskGenerator(sam2)

    def generate_masks(self, image_path):
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        masks = self.mask_generator.generate(image)



def run_example():
    sam_checkpoint = './sam2/checkpoints/sam2.1_hiera_base_plus.pt'
    model_cfg = './sam2/sam2/configs/sam2.1/sam2.1_hiera_b+.yaml'
    segmentation = SAM(sam_checkpoint, model_cfg)

    #sample image
    image_path = './images/groceries.jpg'

    #sample prompt
    text_prompt = 'bag'

    masks = SAM.generate_masks(image_path)


if __name__=='__main__':
    run_example()