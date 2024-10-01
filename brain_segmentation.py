import numpy as np
import cv2
from skimage import filters
from skimage.morphology import remove_small_objects, remove_small_holes

class BrainSegmenter:
    def __init__(self):
        self.preprocessing_steps = []
    
    def preprocess(self, image):

        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        
        return enhanced
    
    def segment(self, image):

        preprocessed = self.preprocess(image)
        

        thresh = filters.threshold_otsu(preprocessed)
        binary = preprocessed > thresh
        
        cleaned = remove_small_objects(binary)
        cleaned = remove_small_holes(cleaned)
        
        # Convert back to uint8
        mask = (cleaned * 255).astype(np.uint8)
        
        return mask

def evaluate_segmentation(pred_mask, true_mask):
    intersection = np.logical_and(pred_mask, true_mask).sum()
    union = np.logical_or(pred_mask, true_mask).sum()
    iou = intersection / union if union != 0 else 0
    dice = 2 * intersection / (pred_mask.sum() + true_mask.sum()) if (pred_mask.sum() + true_mask.sum()) != 0 else 0
    return {'iou': iou, 'dice': dice}

if __name__ == "__main__":

    image = cv2.imread('sample_brain_mri.png')
    

    segmenter = BrainSegmenter()
    mask = segmenter.segment(image)
    

    cv2.imwrite('segmentation_result.png', mask)