import random
import os
from skimage import io
import matplotlib.pyplot as plt
import matplotlib.patches as patches

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