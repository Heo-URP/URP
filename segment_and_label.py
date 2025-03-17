import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from transformers import CLIPProcessor, CLIPModel

# Try different import approaches for SAM 2.1
try:
    # First try direct import for SAM 2.1
    from segment_anything.modeling import Sam2Model
    from segment_anything import SamPredictor
    SAM2_AVAILABLE = True
    print("Using SAM 2.1 via direct Sam2Model import")
except ImportError:
    try:
        # Try segment-anything-fast library
        from segment_anything_fast import sam2_model_registry, SamPredictor
        SAM2_AVAILABLE = True
        print("Using SAM 2.1 via segment-anything-fast")
    except ImportError:
        try:
            # Try updated segment-anything with sam2_model_registry
            from segment_anything import sam2_model_registry, SamPredictor
            SAM2_AVAILABLE = True
            print("Using SAM 2.1 via segment_anything.sam2_model_registry")
        except ImportError:
            # Fall back to original SAM
            from segment_anything import sam_model_registry, SamPredictor
            SAM2_AVAILABLE = False
            print("SAM 2.1 not available, falling back to original SAM")

class SAMCLIPLabeler:
    def __init__(self, sam_checkpoint, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        print(f"Using device: {device}")
        
        # Initialize SAM model
        if SAM2_AVAILABLE:
            print(f"Loading SAM 2.1 model from {sam_checkpoint}...")
            try:
                # Try direct model loading first
                if 'Sam2Model' in globals():
                    self.sam = Sam2Model.from_pretrained("sam2_hiera_l", checkpoint=sam_checkpoint)
                    print("Successfully loaded SAM 2.1 using direct loading!")
                else:
                    # Try registry approach
                    self.sam = sam2_model_registry["sam2_hiera_l"](checkpoint=sam_checkpoint)
                    print("Successfully loaded SAM 2.1 using registry!")
            except Exception as e:
                print(f"Error loading SAM 2.1: {str(e)}")
                print("Falling back to original SAM...")
                # Fallback to original SAM
                self.sam = sam_model_registry["vit_h"](checkpoint=sam_checkpoint)
        else:
            print(f"Loading original SAM model from {sam_checkpoint}...")
            self.sam = sam_model_registry["vit_h"](checkpoint=sam_checkpoint)
            
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
        # Import the appropriate mask generator
        if SAM2_AVAILABLE:
            try:
                from segment_anything import SamAutomaticMaskGenerator
            except ImportError:
                from segment_anything_fast import SamAutomaticMaskGenerator
        else:
            from segment_anything import SamAutomaticMaskGenerator
        
        # Load the image
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Initialize the automatic mask generator with parameters optimized for the model being used
        if SAM2_AVAILABLE:
            mask_generator = SamAutomaticMaskGenerator(
                model=self.sam,
                points_per_side=32,
                pred_iou_thresh=0.88,  # Slightly higher for SAM 2.1
                stability_score_thresh=0.95,
                crop_n_layers=1,
                crop_n_points_downscale_factor=2,
                min_mask_region_area=100,
            )
        else:
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
    
    def label_segments(self, image, masks, candidate_labels, prompt=None, threshold=0.2):
        """
        Label image segments using CLIP and optionally filter by a prompt.
        
        Args:
            image: RGB image as numpy array
            masks: List of binary masks
            candidate_labels: List of text labels to classify against
            prompt: Optional text prompt to filter segments (e.g., "a cat sitting on sofa")
            threshold: Confidence threshold for prompt filtering
            
        Returns:
            List of tuples containing (mask, label, confidence) or 
            (mask, label, confidence, prompt_score) if prompt is provided
        """
        results = []
        
        for mask in masks:
            # Apply mask to image
            masked_image = image.copy()
            masked_image[~mask] = 0
            
            # Convert to PIL Image for CLIP
            masked_pil = Image.fromarray(masked_image)
            
            # Prepare image inputs
            image_inputs = self.clip_processor(
                images=masked_pil,
                return_tensors="pt"
            ).to(self.device)
            
            # Prepare text inputs for candidate labels
            text_inputs = self.clip_processor(
                text=candidate_labels,
                return_tensors="pt",
                padding=True
            ).to(self.device)
            
            # Get CLIP outputs for candidate labels
            with torch.no_grad():
                outputs = self.clip_model(**{
                    **image_inputs,
                    **text_inputs
                })
            
            # Calculate similarity scores for candidate labels
            logits_per_image = outputs.logits_per_image
            probs = logits_per_image.softmax(dim=1)
            
            # Get the best label
            best_idx = probs.argmax().item()
            confidence = probs[0][best_idx].item()
            best_label = candidate_labels[best_idx]
            
            # If prompt is provided, check if this segment matches the prompt
            if prompt:
                # Process the prompt
                prompt_inputs = self.clip_processor(
                    text=[prompt],
                    return_tensors="pt",
                    padding=True
                ).to(self.device)
                
                # Get CLIP outputs for the prompt
                with torch.no_grad():
                    prompt_outputs = self.clip_model(**{
                        **image_inputs,
                        **prompt_inputs
                    })
                
                # Calculate similarity score for the prompt
                prompt_logits = prompt_outputs.logits_per_image
                prompt_score = prompt_logits[0][0].item()
                
                # Only keep segments with high enough similarity to the prompt
                if prompt_score >= threshold:
                    results.append((mask, best_label, confidence, prompt_score))
            else:
                # If no prompt, keep all segments
                results.append((mask, best_label, confidence))
        
        # Sort by prompt score if prompt was provided
        if prompt and results and len(results[0]) > 3:
            results = sorted(results, key=lambda x: x[3], reverse=True)
            
        return results
    
    def visualize_results(self, image, results, output_path=None, show_prompt_score=False):
        """
        Visualize segmentation results.
        
        Args:
            image: RGB image as numpy array
            results: List of tuples from label_segments
            output_path: Path to save visualization (optional)
            show_prompt_score: Whether to show prompt matching score
        """
        plt.figure(figsize=(12, 12))
        plt.imshow(image)
        
        # Create random colors for visualization
        np.random.seed(42)
        colors = [
            np.concatenate([np.random.random(3), [0.35]])
            for _ in range(len(results))
        ]
        
        # Plot each mask with its label
        for i, result in enumerate(results):
            if len(result) > 3:  # Has prompt score
                mask, label, confidence, prompt_score = result
                if show_prompt_score:
                    score_text = f"{label} (match: {prompt_score:.2f})"
                else:
                    score_text = f"{label} ({confidence:.2f})"
            else:
                mask, label, confidence = result
                score_text = f"{label} ({confidence:.2f})"
            
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
                    score_text,
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
    sam_checkpoint = "../checkpoints/sam2.1_hiera_large.pt"  # Path to SAM 2.1 model
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
    
    # Prompt for what you want to segment
    # Set to None if you want to label everything
    segmentation_prompt = input("Enter what to segment (e.g., 'a black cat', 'a person wearing glasses'): ")
    if segmentation_prompt.strip() == "":
        segmentation_prompt = None
    
    # Confidence threshold for prompt matching
    threshold = 0.2  # Lower this value for more inclusive results
    
    # Segmentation method
    method = input("Choose segmentation method (auto/points) [auto]: ").lower() or "auto"
    
    # Process an image
    print(f"Generating masks using {method} method...")
    if method == "auto":
        image, masks = labeler.generate_automatic_masks(image_path)
    else:
        image, masks = labeler.generate_masks_from_points(image_path, num_points=25)
    
    print(f"Found {len(masks)} segments")
    
    # Label the segments with optional prompt filter
    print("Labeling segments with CLIP...")
    if segmentation_prompt:
        print(f"Filtering for segments matching: '{segmentation_prompt}'")
    
    results = labeler.label_segments(image, masks, candidate_labels, segmentation_prompt, threshold)
    
    if len(results) == 0:
        print("No matching segments found. Try a different prompt or lower the threshold.")
    else:
        # Visualize and save the results
        output_path = "segmentation_result.jpg"
        labeler.visualize_results(image, results, output_path, show_prompt_score=segmentation_prompt is not None)
        
        print(f"Processed image with {len(results)} labeled segments")
        for i, result in enumerate(results):
            if segmentation_prompt and len(result) > 3:
                _, label, confidence, prompt_score = result
                print(f"Segment {i+1}: {label} (confidence: {confidence:.2f}, prompt match: {prompt_score:.2f})")
            else:
                _, label, confidence = result
                print(f"Segment {i+1}: {label} (confidence: {confidence:.2f})")


if __name__ == "__main__":
    main()