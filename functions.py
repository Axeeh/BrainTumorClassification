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
    
    
def create_mask(annotations, image_id):
    """
    return the mask of the image

    Parameters:
        annotations (dict): COCO annotations.
        image_id (int): ID of the image to visualize.
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
    rs, cs = np.where(mask_ignored)
    ix = np.random.randint(len(rs), size=size)
    return rs[ix], cs[ix]


def extract_patches(annotations, image_id):

    image_info = next((img for img in annotations['images'] if img['id'] == image_id), None)
    img_path = os.path.join('Dataset/train', image_info['file_name'])
    im = io.imread(img_path)
    
    mask = create_mask(annotations, image_id)    

    patches = []
    labels = []

    # check di charles

    rs,cs = pick_random_centers(mask, size=1000, ignore=32)
    # plt.plot(cs, rs, 'b.')
    for r,c in zip(rs,cs):
        patches.append(im[r-32:r+32, c-32:c+32, :])
        labels.append(mask[r,c])

    rs,cs = pick_random_centers(~mask, size=1000, ignore=32)
    # plt.plot(cs, rs, 'w.')
    for r,c in zip(rs,cs):
        patches.append(im[r-32:r+32, c-32:c+32, :])
        labels.append(mask[r,c])
        
    return patches, labels