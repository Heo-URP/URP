import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from transformers import CLIPProcessor, CLIPModel
from segment_anything import sam_model_registry, SamPredictor

class SAMCLIPLabeler:
    def __init__(self, sam_checkpoint, device="cuda" if torch.cuda.is_available() else "cpu"):

        self.device = device
        print(f"Using device: {device}")
        
        # Initialize SAM
        print("Loading SAM model...")
        self.sam = sam_model_registry["vit_b"](checkpoint=sam_checkpoint)
        self.sam.to(device=self.device)
        self.predictor = SamPredictor(self.sam)
        
        # Initialize CLIP
        print("Loading CLIP model...")
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_model.to(device=self.device)
        print("Models loaded successfully!")
        
    def generate_masks_from_points(self, image_path, num_points=16):

        # Load the image
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Set the image in the predictor
        self.predictor.set_image(image)
        
        # Generate grid of points
        h, w = image.shape[:2]
        rows = cols = int(np.sqrt(num_points))
        
        y_points = np.linspace(h//rows//2, h-h//rows//2, rows).astype(int)
        x_points = np.linspace(w//cols//2, w-w//cols//2, cols).astype(int)
        
        points = []
        for y in y_points:
            for x in x_points:
                points.append([x, y])
        
        # Generate masks for each point
        all_masks = []
        for point in points:
            input_point = np.array([point])
            input_label = np.array([1])  # 1 for foreground
            
            try:
                masks, scores, _ = self.predictor.predict(
                    point_coords=input_point,
                    point_labels=input_label,
                    multimask_output=True
                )
                
                # Keep the mask with the highest score
                if len(scores) > 0:
                    best_mask_idx = np.argmax(scores)
                    all_masks.append(masks[best_mask_idx])
            except Exception as e:
                print(f"Error generating mask for point {point}: {e}")
        
        # Filter small masks
        all_masks = [mask for mask in all_masks if mask.sum() > 100]  # Minimum 100 pixels
        
        # Non-maximum suppression to remove overlapping masks
        filtered_masks = self.non_max_suppression(all_masks, threshold=0.5)
        
        return image, filtered_masks
    
    def generate_automatic_masks(self, image_path):

        from segment_anything import SamAutomaticMaskGenerator
        
        # Load the image
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Initialize the automatic mask generator
        mask_generator = SamAutomaticMaskGenerator(
            model=self.sam,
            points_per_side=32,
            pred_iou_thresh=0.86,
            stability_score_thresh=0.92,
            crop_n_layers=1,
            crop_n_points_downscale_factor=2,
            min_mask_region_area=100,
        )
        
        # Generate masks
        masks = mask_generator.generate(image)
        
        # Convert to binary masks
        binary_masks = []
        for mask_data in masks:
            binary_mask = mask_data['segmentation']
            binary_masks.append(binary_mask)
        
        return image, binary_masks
    
    def non_max_suppression(self, masks, threshold=0.5):

        if not masks:
            return []
            
        # Calculate areas for each mask
        areas = [np.sum(mask) for mask in masks]
        
        # Sort masks by area (largest first)
        order = np.argsort(areas)[::-1]
        
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            
            # Calculate IoU with other masks
            ious = []
            for j in order[1:]:
                intersection = np.logical_and(masks[i], masks[j]).sum()
                union = np.logical_or(masks[i], masks[j]).sum()
                iou = intersection / union if union > 0 else 0
                ious.append(iou)
            
            # Keep masks with IoU below threshold
            if not ious:
                break
            inds = np.where(np.array(ious) <= threshold)[0]
            order = order[inds + 1]
        
        return [masks[i] for i in keep]
    
    def label_segments(self, image, masks, candidate_labels):

        results = []
        
        for mask in masks:
            # Apply mask to image
            masked_image = image.copy()
            masked_image[~mask] = 0
            
            # Convert to PIL Image for CLIP
            masked_pil = Image.fromarray(masked_image)
            
            # Prepare text inputs
            text_inputs = self.clip_processor(
                text=candidate_labels,
                return_tensors="pt",
                padding=True
            ).to(self.device)
            
            # Prepare image inputs
            image_inputs = self.clip_processor(
                images=masked_pil,
                return_tensors="pt"
            ).to(self.device)
            
            # Get CLIP outputs
            with torch.no_grad():
                outputs = self.clip_model(**{
                    **image_inputs,
                    **text_inputs
                })
            
            # Calculate similarity scores
            logits_per_image = outputs.logits_per_image
            probs = logits_per_image.softmax(dim=1)
            
            # Get the best label
            best_idx = probs.argmax().item()
            confidence = probs[0][best_idx].item()
            best_label = candidate_labels[best_idx]
            
            results.append((mask, best_label, confidence))
        
        return results
    
    def visualize_results(self, image, results, output_path=None):

        plt.figure(figsize=(12, 12))
        plt.imshow(image)
        
        # Create random colors for visualization
        np.random.seed(42)
        colors = [
            np.concatenate([np.random.random(3), [0.35]])
            for _ in range(len(results))
        ]
        
        # Plot each mask with its label
        for i, (mask, label, confidence) in enumerate(results):
            colored_mask = np.ones((mask.shape[0], mask.shape[1], 4))
            colored_mask[:, :, :3] = colors[i][:3]
            colored_mask[:, :, 3] = mask * colors[i][3]
            
            plt.imshow(colored_mask)
            
            y_indices, x_indices = np.where(mask)
            if len(y_indices) > 0 and len(x_indices) > 0:
                x_center = np.mean(x_indices)
                y_center = np.mean(y_indices)
                plt.text(
                    x_center, y_center, 
                    f"{label} ({confidence:.2f})",
                    color='white', fontsize=12, 
                    bbox=dict(facecolor='black', alpha=0.5)
                )
        
        plt.axis('off')
        if output_path:
            plt.savefig(output_path, bbox_inches='tight')
            print(f"Visualization saved to {output_path}")
        else:
            plt.show()


# Example usage
def main():
    # Initialize the model
    sam_checkpoint = "./models/sam_vit_b_01ec64.pth"  # Replace with actual path to the original SAM checkpoint
    labeler = SAMCLIPLabeler(sam_checkpoint)
    
    # Define candidate labels (you can expand this list)
    candidate_labels = [
        "person", "cat", "dog", "car", "bicycle", "building", "tree", "sky",
        "grass", "flower", "bird", "laptop", "phone", "table", "chair", "food",
        "water", "mountain", "beach", "road", "book", "clothing", "hat", "cup",
        "bottle", "sofa", "television", "airplane", "train", "boat", "wall", "window"
    ]
    
    # Path to your image
    image_path = "./images/cat1.jpg" 
    
    # Process an image - choose one of the two methods:
    
    # Method 1: Using point-based segmentation
    print("Generating masks from points...")
    image, masks = labeler.generate_masks_from_points(image_path, num_points=25)
    
    
    print(f"Found {len(masks)} segments")
    
    # Label the segments
    print("Labeling segments with CLIP...")
    results = labeler.label_segments(image, masks, candidate_labels)
    
    # Visualize and save the results
    output_path = "segmentation_result.jpg"
    labeler.visualize_results(image, results, output_path)
    
    print(f"Processed image with {len(results)} labeled segments")
    for i, (_, label, confidence) in enumerate(results):
        print(f"Segment {i+1}: {label} (confidence: {confidence:.2f})")


if __name__ == "__main__":
    main()