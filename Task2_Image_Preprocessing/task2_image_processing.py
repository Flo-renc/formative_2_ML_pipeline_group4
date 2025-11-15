# Task 2 - Image Preprocessing
# Completed by: MarialRK
# Date: [Add today's date]

import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def load_images_from_subfolders(image_dir):
    """
    Load all images from subfolders (Daniel, Florence, David)
    Returns: image_paths, member_names
    """
    subfolders = [f for f in os.listdir(image_dir) if os.path.isdir(os.path.join(image_dir, f))]
    image_paths = []
    member_names = []
    
    print(f"Found {len(subfolders)} team member folders: {subfolders}")
    
    for folder in subfolders:
        folder_path = os.path.join(image_dir, folder)
        files_in_folder = os.listdir(folder_path)
        
        for file in files_in_folder:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_paths.append(os.path.join(folder_path, file))
                member_names.append(folder)
    
    print(f"Total images loaded: {len(image_paths)}")
    return image_paths, member_names

def display_images_grid(image_paths, member_names):
    """Display all images in a grid layout"""
    num_images = len(image_paths)
    if num_images == 0:
        print("No images to display")
        return
        
    rows = (num_images + 2) // 3
    fig, axes = plt.subplots(rows, 3, figsize=(15, 5*rows))
    
    if rows == 1:
        axes = axes.reshape(1, -1)
    
    for i, (img_path, member) in enumerate(zip(image_paths, member_names)):
        img = cv2.imread(img_path)
        if img is not None:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            row = i // 3
            col = i % 3
            axes[row, col].imshow(img_rgb)
            axes[row, col].set_title(f"{member}\n{os.path.basename(img_path)}")
            axes[row, col].axis('off')
    
    # Hide empty subplots
    for i in range(len(image_paths), rows*3):
        row = i // 3
        col = i % 3
        axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.show()

def rotate_image(image, angle=15):
    """Rotate image by specified angle"""
    height, width = image.shape[:2]
    rotation_matrix = cv2.getRotationMatrix2D((width/2, height/2), angle, 1)
    rotated = cv2.warpAffine(image, rotation_matrix, (width, height))
    return rotated

def flip_image(image):
    """Flip image horizontally"""
    return cv2.flip(image, 1)

def augment_images(original_images, image_labels):
    """Apply augmentations to all images"""
    augmented_images = []
    augmented_labels = []
    
    for img, label in zip(original_images, image_labels):
        # Keep original
        augmented_images.append(img)
        augmented_labels.append(f"{label}_original")
        
        # Apply rotations
        rotated1 = rotate_image(img, 15)
        augmented_images.append(rotated1)
        augmented_labels.append(f"{label}_rotated15")
        
        rotated2 = rotate_image(img, -15)
        augmented_images.append(rotated2)
        augmented_labels.append(f"{label}_rotated-15")
        
        # Apply flipping
        flipped = flip_image(img)
        augmented_images.append(flipped)
        augmented_labels.append(f"{label}_flipped")
    
    print(f"Created {len(augmented_images)} images from {len(original_images)} originals")
    return augmented_images, augmented_labels

def extract_hog_features(image):
    """Extract HOG features from image"""
    resized = cv2.resize(image, (128, 128))
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    hog = cv2.HOGDescriptor((128, 128), (16, 16), (8, 8), (8, 8), 9)
    hog_features = hog.compute(gray)
    return hog_features.flatten()

def extract_color_histogram(image, bins=32):
    """Extract color histogram features"""
    hist_b = cv2.calcHist([image], [0], None, [bins], [0, 256])
    hist_g = cv2.calcHist([image], [1], None, [bins], [0, 256])
    hist_r = cv2.calcHist([image], [2], None, [bins], [0, 256])
    cv2.normalize(hist_b, hist_b)
    cv2.normalize(hist_g, hist_g)
    cv2.normalize(hist_r, hist_r)
    return np.concatenate([hist_b.flatten(), hist_g.flatten(), hist_r.flatten()])

def extract_features_batch(images):
    """Extract features from all images"""
    print("Extracting features from images...")
    all_features = []
    
    for i, img in enumerate(images):
        hog_features = extract_hog_features(img)
        color_features = extract_color_histogram(img)
        combined_features = np.concatenate([hog_features, color_features])
        all_features.append(combined_features)
        
        if (i + 1) % 10 == 0:
            print(f"Processed {i + 1}/{len(images)} images")
    
    print("Feature extraction completed!")
    return all_features

def create_features_dataframe(features, labels, member_names):
    """Create DataFrame with all features and metadata"""
    feature_columns = [f'feature_{i}' for i in range(len(features[0]))]
    
    # Parse labels to get expressions and augmentation types
    expressions = []
    augmentation_types = []
    parsed_members = []
    
    for label in labels:
        parts = label.split('_')
        parsed_members.append(parts[0])
        expressions.append(parts[1] if len(parts) > 2 else 'unknown')
        augmentation_types.append(parts[-1])
    
    # Create DataFrame
    df = pd.DataFrame(features, columns=feature_columns)
    df['image_id'] = labels
    df['member'] = parsed_members
    df['expression'] = expressions
    df['augmentation_type'] = augmentation_types
    
    # Reorder columns
    cols = ['image_id', 'member', 'expression', 'augmentation_type'] + feature_columns
    df = df[cols]
    
    return df

# Main execution function
def process_image_pipeline(image_dir, output_csv_path):
    """Complete image processing pipeline"""
    print("=== Starting Image Processing Pipeline ===")
    
    # 1. Load images
    image_paths, member_names = load_images_from_subfolders(image_dir)
    
    if len(image_paths) == 0:
        print("No images found! Exiting pipeline.")
        return None
    
    # 2. Display images
    display_images_grid(image_paths, member_names)
    
    # 3. Load original images
    original_images = []
    original_labels = []
    
    for img_path, member in zip(image_paths, member_names):
        img = cv2.imread(img_path)
        if img is not None:
            original_images.append(img)
            filename = os.path.splitext(os.path.basename(img_path))[0]
            original_labels.append(f"{member}_{filename}")
    
    # 4. Apply augmentations
    augmented_images, augmented_labels = augment_images(original_images, original_labels)
    
    # 5. Extract features
    all_features = extract_features_batch(augmented_images)
    
    # 6. Create DataFrame
    df_features = create_features_dataframe(all_features, augmented_labels, member_names)
    
    # 7. Save to CSV
    df_features.to_csv(output_csv_path, index=False)
    print(f"✅ Features saved to: {output_csv_path}")
    print(f"📊 DataFrame shape: {df_features.shape}")
    
    return df_features

if __name__ == "__main__":
    # Example usage
    image_directory = "/content/drive/MyDrive/FaceProject/images/"
    output_csv = "image_features.csv"
    
    df = process_image_pipeline(image_directory, output_csv)
