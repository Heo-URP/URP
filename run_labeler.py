import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from transformers import CLIPProcessor, CLIPModel
from transformers import AutoFeatureExtractor, AutoModelForImageSegmentation
import random

class SegmentationLabeler:
    def __init__(self, device="cuda" if torch.cuda.is_available() else "cpu"):
        """
        Initialize the segmentation and CLIP models from HuggingFace.
        """
        self.device = device
        print(f"Using device: {device}")
        
        # Initialize segmentation model (using Mask2Former which is easier to install)
        print("Loading segmentation model...")
        self.seg_extractor = AutoFeatureExtractor.from_pretrained("facebook/mask2former-swin-tiny-coco-instance")
        self.seg_model = AutoModelForImageSegmentation.from_pretrained("facebook/mask2former-swin-tiny-coco-instance")
        self.seg_model.to(device=self.device)
        
        # Initialize CLIP
        print("Loading CLIP model...")
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_model.to(device=self.device)
        print("Models loaded successfully!")
        
    def segment_image(self, image_path, score_threshold=0.85):
        """
        Segment an image using a pretrained segmentation model.
        
        Args:
            image_path: Path to the image
            score_threshold: Minimum confidence score for segments
            
        Returns:
            original_image: The original image
            masks: List of segmentation masks
        """
        # Load the image
        image = Image.open(image_path)
        image_array = np.array(image)
        
        # Process the image for segmentation
        inputs = self.seg_extractor(images=image, return_tensors="pt").to(self.device)
        outputs = self.seg_model(**inputs)
        
        # Process results
        target_sizes = torch.tensor([image_array.shape[:2]])
        results = self.seg_extractor.post_process_semantic_segmentation(
            outputs, target_sizes=target_sizes
        )[0]
        
        # Extract masks with high enough scores
        masks = []
        for mask in results.cpu().numpy():
            # Convert to boolean mask
            bool_mask = mask.astype(bool)
            if bool_mask.sum() > 100:  # Only keep masks with at least 100 pixels
                masks.append(bool_mask)
        
        # Apply non-maximum suppression to remove overlapping masks
        filtered_masks = self.non_max_suppression(masks, threshold=0.5)
        
        return image_array, filtered_masks
    
    def non_max_suppression(self, masks, threshold=0.5):
        """
        Apply non-maximum suppression to remove highly overlapping masks.
        
        Args:
            masks: List of masks
            threshold: IoU threshold for suppression
            
        Returns:
            filtered_masks: List of filtered masks
        """
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
        """
        Label each segment using CLIP.
        
        Args:
            image: Original image
            masks: List of segmentation masks
            candidate_labels: List of candidate labels to match against
            
        Returns:
            results: List of (mask, label, confidence) tuples
        """
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
        """
        Visualize the segmentation and labeling results.
        
        Args:
            image: Original image
            results: List of (mask, label, confidence) tuples
            output_path: Path to save the visualization
            
        Returns:
            None (displays or saves the visualization)
        """
        plt.figure(figsize=(12, 12))
        plt.imshow(image)
        
        # Create random colors for visualization
        random.seed(42)
        colors = [
            (random.random(), random.random(), random.random(), 0.4)
            for _ in range(len(results))
        ]
        
        # Plot each mask with its label
        for i, (mask, label, confidence) in enumerate(results):
            # Create colored mask
            colored_mask = np.zeros((mask.shape[0], mask.shape[1], 4), dtype=np.float32)
            colored_mask[:, :, 0] = colors[i][0]
            colored_mask[:, :, 1] = colors[i][1]
            colored_mask[:, :, 2] = colors[i][2]
            colored_mask[:, :, 3] = mask * colors[i][3]
            
            plt.imshow(colored_mask)
            
            # Find the center of the mask for text placement
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


def main():
    """
    Main function to demonstrate the segmentation and labeling process.
    """
    # Initialize the segmentation and labeling model
    print("Initializing models...")
    labeler = SegmentationLabeler()
    
    # Define candidate labels (you can expand this list)
    candidate_labels = [
        "person", "cat", "dog", "car", "bicycle", "building", "tree", "sky",
        "grass", "flower", "bird", "laptop", "phone", "table", "chair", "food",
        "water", "mountain", "beach", "road", "book", "clothing", "hat", "cup",
        "bottle", "sofa", "television", "airplane", "train", "boat", "traffic light",
        "fire hydrant", "stop sign", "parking meter", "bench", "backpack", "umbrella",
        "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
        "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", 
        "tennis racket", "window", "door", "plant", "fruit", "vegetable"
    ]
    
    # Path to your image
    image_path = "./images/cat1.jpg"  # Replace with your image path
    
    # Process the image
    print(f"Processing image: {image_path}")
    image, masks = labeler.segment_image(image_path)
    print(f"Found {len(masks)} segments")
    
    # Label the segments
    print("Labeling segments with CLIP...")
    results = labeler.label_segments(image, masks, candidate_labels)
    
    # Visualize and save the results
    output_path = "segmentation_result.jpg"
    print("Visualizing results...")
    labeler.visualize_results(image, results, output_path)
    
    # Print results
    print("\nResults:")
    for i, (_, label, confidence) in enumerate(results):
        print(f"Segment {i+1}: {label} (confidence: {confidence:.2f})")


if __name__ == "__main__":
    main()