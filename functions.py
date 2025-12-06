import pandas as pd
import matplotlib.pyplot as plt
import random
import matplotlib.patches as patches
from skimage import io
import os
import numpy as np

def display_images_by_category(annotations, category_id, num_images=5):
    # Filter annotations for the given category_id
    filtered_annotations = [ann for ann in annotations['annotations'] if ann['category_id'] == category_id]

    # Get unique image IDs for the filtered annotations
    image_ids = list(set(ann['image_id'] for ann in filtered_annotations))

    # Randomly select a subset of image IDs
    selected_image_ids = random.sample(image_ids, min(num_images, len(image_ids)))

    # Get image metadata for the selected image IDs
    selected_images = [img for img in annotations['images'] if img['id'] in selected_image_ids]

    # Display images with annotations
    plt.figure(figsize=(15, 10))
    for i, img_info in enumerate(selected_images):
        img_path = os.path.join('Dataset/train', img_info['file_name'])
        img = io.imread(img_path)

        # Get annotations for the current image
        img_annotations = [ann for ann in filtered_annotations if ann['image_id'] == img_info['id']]

        # Plot the image
        ax = plt.subplot(2, 3, i + 1)
        ax.imshow(img)
        ax.axis('off')
        ax.set_title(f"Image ID: {img_info['id']}")

        # Add annotations (bounding boxes)
        for ann in img_annotations:
            bbox = ann['bbox']
            rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], linewidth=2, edgecolor='r', facecolor='none')
            ax.add_patch(rect)

    plt.tight_layout()
    plt.show()


def display_image_with_annotations(ax, image, annotations, display_type='both', colors=None):
    ax.imshow(image)
    ax.axis('off')  # Turn off the axes

    # Define a default color map if none is provided
    if colors is None:
        colors = plt.cm.tab10

    for ann in annotations:
        category_id = ann['category_id']
        color = colors(category_id % 10)
        
        # Display bounding box
        if display_type in ['bbox', 'both']:
            bbox = ann['bbox']
            # print(bbox)
            rect = patches.CirclePolygon ((bbox[0], bbox[1]), bbox[2], bbox[3], linewidth=1, edgecolor=color, facecolor='none')
            ax.add_patch(rect)
        
        # Display segmentation polygon
        if display_type in ['seg', 'both']:
            for seg in ann['segmentation']:
                poly = [(seg[i], seg[i+1]) for i in range(0, len(seg), 2)]
                # print(poly)
                polygon = patches.Polygon(poly, closed=True, edgecolor=color, fill=False)
                ax.add_patch(polygon)

def display_images_with_coco_annotations(image_paths, annotations, display_type='both', colors=None):
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))

    for ax, img_path in zip(axs.ravel(), image_paths):
        
        image = io.imread(img_path)

        # Get image filename to match with annotations
        img_filename = os.path.basename(img_path)
        img_id = next(item for item in annotations['images'] if item["file_name"] == img_filename)['id']
        
        # Filter annotations for the current image
        img_annotations = [ann for ann in annotations['annotations'] if ann['image_id'] == img_id]
        
        display_image_with_annotations(ax, image, img_annotations, display_type, colors)

    plt.tight_layout()
    plt.show()
    
    
    
def visualize_annotation_mask(annotations, image_id, show_annotations=False):
    """
    Visualize the annotation mask for a given image ID, with an option to display annotations.

    Parameters:
        annotations (dict): COCO annotations.
        image_id (int): ID of the image to visualize.
        show_annotations (bool): Whether to display annotations on the image.
    """
    # Find the image metadata
    image_info = next((img for img in annotations['images'] if img['id'] == image_id), None)

    # Load the image
    img_path = os.path.join('Dataset/train', image_info['file_name'])
    image = io.imread(img_path)

    # Create an empty mask
    mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)

    # Add annotations to the mask
    for ann in annotations['annotations']:
        if ann['image_id'] == image_id:
            for seg in ann.get('segmentation', []):
                poly = np.array(seg).reshape(-1, 2)
                x_min, y_min = np.min(poly, axis=0).astype(int)
                x_max, y_max = np.max(poly, axis=0).astype(int)

                for y in range(y_min, y_max + 1):
                    for x in range(x_min, x_max + 1):
                        # Simple point-in-polygon check
                        n = len(poly)
                        inside = False
                        px, py = x, y
                        for i in range(n):
                            x1, y1 = poly[i]
                            x2, y2 = poly[(i + 1) % n]
                            if ((y1 > py) != (y2 > py)) and (px < (x2 - x1) * (py - y1) / (y2 - y1) + x1):
                                inside = not inside
                        if inside:
                            mask[y, x] = 255

    # Visualize the image and mask
    plt.figure(figsize=(15, 5))


    # Annotation mask
    plt.subplot(1, 3 if show_annotations else 2, 2)
    plt.imshow(mask, cmap='gray')
    plt.title("Annotation Mask")
    plt.axis('off')

    # Optional: Display annotations
    if show_annotations:
        plt.subplot(1, 3, 3)
        plt.imshow(image)
        plt.title("Image with Annotations")
        plt.axis('off')

        for ann in annotations['annotations']:
            if ann['image_id'] == image_id:
                for seg in ann.get('segmentation', []):
                    poly = np.array(seg).reshape(-1, 2)
                    polygon = patches.Polygon(poly, closed=True, edgecolor='r', fill=False, linewidth=2)
                    plt.gca().add_patch(polygon)
    else:
        plt.subplot(1, 2, 1)
        plt.imshow(image)
        plt.title("Original Image")
        plt.axis('off')

    # Adjust layout for better visualization
    plt.tight_layout()
    plt.show()    
    
    
def create_mask(annotations, image_id, where = "train"):
    """
    return the mask of the image

    Parameters:
        annotations (dict): COCO annotations.
        image_id (int): ID of the image to visualize.
    """
    # Find the image metadata
    image_info = next((img for img in annotations['images'] if img['id'] == image_id), None)

    # Load the image
    img_path = os.path.join(f'Dataset/{where}', image_info['file_name'])
    image = io.imread(img_path)

    # Create an empty mask
    mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)

    # Add annotations to the mask
    for ann in annotations['annotations']:
        if ann['image_id'] == image_id:
            for seg in ann.get('segmentation', []):
                poly = np.array(seg).reshape(-1, 2)
                x_min, y_min = np.min(poly, axis=0).astype(int)
                x_max, y_max = np.max(poly, axis=0).astype(int)

                for y in range(y_min, y_max + 1):
                    for x in range(x_min, x_max + 1):
                        # Simple point-in-polygon check
                        n = len(poly)
                        inside = False
                        px, py = x, y
                        for i in range(n):
                            x1, y1 = poly[i]
                            x2, y2 = poly[(i + 1) % n]
                            if ((y1 > py) != (y2 > py)) and (px < (x2 - x1) * (py - y1) / (y2 - y1) + x1):
                                inside = not inside
                        if inside:
                            mask[y, x] = 255

    return mask


def sample_annotation_mask_pixels(mask, num_true=3, num_false=3, random_state=None):
    """
    Sample random pixel coordinates from the annotation mask.

    Returns up to ⁠ num_true ⁠ coordinates where the mask is True and
    ⁠ num_false ⁠ coordinates where the mask is False.
    """
    rng = np.random.default_rng(random_state)
    mask_bool = mask.astype(bool)

    true_positions = np.argwhere(mask_bool)
    false_positions = np.argwhere(~mask_bool)

    if true_positions.shape[0] < num_true:
        raise ValueError("Not enough true pixels to sample")
    if false_positions.shape[0] < num_false:
        raise ValueError("Not enough false pixels to sample")

    true_samples = true_positions[rng.choice(len(true_positions), size=num_true, replace=False)]
    false_samples = false_positions[rng.choice(len(false_positions), size=num_false, replace=False)]

    return {"true": [tuple(p) for p in true_samples], "false": [tuple(p) for p in false_samples]}



# FOR PATCHES
def pick_random_centers(mask, size=100, ignore=0):
    mask_ignored = mask.copy()
    mask_ignored[:ignore,:]=False
    mask_ignored[-ignore:,:]=False
    mask_ignored[:,:ignore]=False
    mask_ignored[:,-ignore:]=False
    
    h,w = mask_ignored.shape
    valid_centers = []
    
    for r in range(ignore, h - ignore):
        for c in range(ignore, w - ignore):
            if mask_ignored[r, c]:
                # Verifica che tutti i pixel nella patch siano True
                patch = mask_ignored[r-ignore:r+ignore, c-ignore:c+ignore]
                if np.all(patch):
                    valid_centers.append((r, c))

    # STEP 3: Seleziona random tra i centri validi
    if len(valid_centers) == 0:
        raise ValueError(f"No valid centers found with ignore={ignore}. Try reducing ignore value.")
    
        
    actual_size = min(size, len(valid_centers))
    selected_indices = np.random.choice(len(valid_centers), size=actual_size, replace=False)
    selected_centers = [valid_centers[i] for i in selected_indices]
    
    
    rs = [c[0] for c in selected_centers]
    cs = [c[1] for c in selected_centers]
    
    return np.array(rs), np.array(cs)


def extract_patches(annotations, image_id, where = "train"):

    image_info = next((img for img in annotations['images'] if img['id'] == image_id), None)
    img_path = os.path.join(f'Dataset/{where}', image_info['file_name'])
    im = io.imread(img_path)
    
    mask = create_mask(annotations, image_id, where)    
    mask_bool = mask.astype(bool)


    patches = []
    labels = []

    # Estrai patches tumorali (garantite al 100% dentro la regione tumor)
    tumor_success = False
    for attempt in range(5):
        try:
            rs, cs = pick_random_centers(mask_bool, size=1, ignore=32)
            for r, c in zip(rs, cs):
                patches.append(im[r-32:r+32, c-32:c+32, :])
                labels.append(255)  # tumor label
            tumor_success = True
            break
        except ValueError as e:
            print(f"Attempt {attempt+1}/5 for tumor patches failed: {e}")
    if not tumor_success:
        print("Warning: failed to extract tumor patches after 5 attempts")

    # Estrai patches non-tumorali (garantite al 100% fuori dalla regione tumor)
    nontumor_success = False
    for attempt in range(5):
        try:
            rs, cs = pick_random_centers(~mask_bool, size=1, ignore=32)
            for r, c in zip(rs, cs):
                patches.append(im[r-32:r+32, c-32:c+32, :])
                labels.append(0)  # non-tumor label
            nontumor_success = True
            break
        except ValueError as e:
            print(f"Attempt {attempt+1}/5 for non-tumor patches failed: {e}")
    if not nontumor_success:
        print("Warning: failed to extract non-tumor patches after 5 attempts")
        
    return patches, labels


# ===== NUOVE FUNZIONI PER UNET SEGMENTATION =====

def get_annotations_dict(coco_data):
    """Crea mapping: image_id -> lista di bounding boxes"""
    annotations_by_id = {}
    for ann in coco_data['annotations']:
        img_id = ann['image_id']
        if img_id not in annotations_by_id:
            annotations_by_id[img_id] = []
        annotations_by_id[img_id].append(ann['bbox'])
    return annotations_by_id


def create_mask_from_bbox(image_shape, bboxes):
    """Crea maschera binaria dai bounding boxes COCO"""
    mask = np.zeros(image_shape, dtype=np.float32)
    for bbox in bboxes:
        x, y, w, h = [int(v) for v in bbox]
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(image_shape[1], x + w), min(image_shape[0], y + h)
        mask[y1:y2, x1:x2] = 1
    return mask


def generate_segmentation_dataset(coco_data, dataset_dir, target_size=64):
    """
    Genera dataset di immagini e maschere per UNet
    - Carica immagine
    - La ridimensiona a target_size
    - Riscala le coordinate della segmentazione
    - Crea la maschera a target_size
    
    Returns:
        X: array (N, target_size, target_size, 3)
        y: array (N, target_size, target_size, 1)
    """
    from PIL import Image
    from skimage import draw
    
    X = []
    y = []
    
    for img_info in coco_data['images']:
        img_id = img_info['id']
        img_path = os.path.join(dataset_dir, img_info['file_name'])
        
        try:
            # Carica immagine a risoluzione originale
            image = Image.open(img_path).convert('RGB')
            original_w, original_h = image.size
            
            # Ridimensiona immagine
            image = image.resize((target_size, target_size), Image.BILINEAR)
            image_array = np.array(image, dtype=np.float32) / 255.0
            
            # Crea maschera a target_size VUOTA
            mask = np.zeros((target_size, target_size), dtype=np.float32)
            
            # Rasterizza le segmentazioni ridimensionate
            for ann in coco_data['annotations']:
                if ann['image_id'] == img_id:
                    for seg in ann.get('segmentation', []):
                        if isinstance(seg, list) and len(seg) >= 6:
                            # Converti lista di coordinate in array
                            poly = np.array(seg, dtype=np.float32).reshape(-1, 2)
                            
                            # Riscala coordinate alla nuova risoluzione
                            poly[:, 0] = poly[:, 0] * (target_size / original_w)
                            poly[:, 1] = poly[:, 1] * (target_size / original_h)
                            
                            # Rasterizza il poligono sulla maschera
                            try:
                                poly_int = poly.astype(np.int32)
                                # draw.polygon(row_coords, col_coords, shape)
                                rr, cc = draw.polygon(poly_int[:, 1], poly_int[:, 0], 
                                                     shape=(target_size, target_size))
                                mask[rr, cc] = 1.0
                            except Exception as e:
                                print(f"Error rasterizing polygon: {e}")
                                continue
            
            X.append(image_array)
            y.append(mask)
            
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            continue
    
    X = np.array(X)  # Shape: (N, target_size, target_size, 3)
    y = np.array(y)[..., np.newaxis]  # Shape: (N, target_size, target_size, 1)
    
    return X, y


def weighted_binary_crossentropy(tf, y_true, y_pred, tumor_weight, non_tumor_weight):
    """Custom loss that weights tumor pixels more heavily"""
    # Clip predictions to avoid log(0)
    y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)
    
    # Binary crossentropy
    bce = -(y_true * tf.math.log(y_pred) + (1 - y_true) * tf.math.log(1 - y_pred))
    
    # Apply weights: tumor pixels get higher weight
    weighted = bce * (y_true * tumor_weight + (1 - y_true) * non_tumor_weight)
    
    return tf.reduce_mean(weighted)


def evaluate_segmentation(model, images, masks, threshold=0.5):
    """Calculate segmentation metrics"""
    predictions = model.predict(images, verbose=0)
    pred_binary = (predictions > threshold).astype(np.uint8)
    
    masks_flat = masks.flatten()
    pred_flat = pred_binary.flatten()
    
    # Accuracy
    accuracy = np.sum(masks_flat == pred_flat) / len(masks_flat)
    
    # Dice coefficient (F1 for segmentation)
    intersection = np.sum(masks_flat * pred_flat)
    dice = 2 * intersection / (np.sum(masks_flat) + np.sum(pred_flat) + 1e-7)
    
    # IoU (Intersection over Union)
    union = np.sum((masks_flat + pred_flat) > 0)
    iou = intersection / union if union > 0 else 0
    
    # Precision and Recall
    tp = np.sum((pred_flat == 1) & (masks_flat == 1))
    fp = np.sum((pred_flat == 1) & (masks_flat == 0))
    fn = np.sum((pred_flat == 0) & (masks_flat == 1))
    
    precision = tp / (tp + fp + 1e-7)
    recall = tp / (tp + fn + 1e-7)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-7)
    
    return {
        'accuracy': accuracy,
        'dice': dice,
        'iou': iou,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


def segment_full_image(classifier, image, patch_size=64, stride=32):
    """
    Applica il classificatore di patch su un'immagine intera usando sliding window
    
    Args:
        classifier: Classificatore addestrato (logistic/cnn/fcnn)
        image: Immagine completa normalizzata [0,1], shape (H, W, 3)
        patch_size: Dimensione patch (64)
        stride: Passo della sliding window
    
    Returns:
        Mappa di probabilità (H, W)
    """
    h, w = image.shape[:2]
    prob_map = np.zeros((h, w), dtype=np.float32)
    count_map = np.zeros((h, w), dtype=np.float32)
    
    patches = []
    positions = []
    
    # Estrai tutte le patch con sliding window
    for y in range(0, h - patch_size + 1, stride):
        for x in range(0, w - patch_size + 1, stride):
            patch = image[y:y+patch_size, x:x+patch_size]
            
            if patch.shape[:2] == (patch_size, patch_size):
                patches.append(patch)
                positions.append((y, x))
    
    if len(patches) == 0:
        return prob_map
    
    patches_array = np.array(patches)
    
    # Predici in batch per efficienza
    if hasattr(classifier, 'predict_proba'):  # Logistic Regression
        patches_flat = patches_array.reshape(len(patches), -1)
        probs = classifier.predict_proba(patches_flat)[:, 1]
    else:  # CNN models
        probs = classifier.predict(patches_array, verbose=0).flatten()
    
    # Posiziona le predizioni nella mappa
    for (y, x), prob in zip(positions, probs):
        prob_map[y:y+patch_size, x:x+patch_size] += prob
        count_map[y:y+patch_size, x:x+patch_size] += 1
    
    # Media delle predizioni sovrapposte
    prob_map = np.divide(prob_map, count_map, where=count_map > 0)
    
    return prob_map