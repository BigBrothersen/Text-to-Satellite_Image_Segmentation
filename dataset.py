import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import numpy as np
import matplotlib.pyplot as plt
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader, random_split

# Default Values (Change depending on the dataset)
COLOUR_MAPPING = {
    (0, 255, 255): 0,    # Urban (Cyan)
    (255, 255, 0): 1,    # Agriculture (Yellow)
    (255, 0, 255): 2,    # Rangeland (Magenta)
    (0, 255, 0): 3,      # Forest (Green)
    (0, 0, 255): 4,      # Water (Blue)
    (255, 255, 255): 5,  # Barren (White)
    (0, 0, 0): 6         # Unknown (Black)
}

LABELS_MAPPING = {
    0: 'Urban',
    1: 'Agriculture',
    2: 'Rangeland',
    3: 'Forest',
    4: 'Water',
    5: 'Barren',
    6: 'Unknown'
}

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

class SatelliteDataset(Dataset):
    def __init__(self, image_dir, mask_dir, colour_mapping=COLOUR_MAPPING, label_mapping=LABELS_MAPPING, transform=None, mean=mean, std=std):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = os.listdir(image_dir)
        self.colour_mapping = colour_mapping
        self.label_mapping = label_mapping
        self.mean = mean
        self.std = std
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        try:
            img_path = os.path.join(self.image_dir, self.images[index])
            mask_rgb = os.path.join(self.mask_dir, self.images[index].replace("_sat.jpg", "_mask.png"))

            image = np.array(Image.open(img_path).convert("RGB"))
            mask_rgb = np.array(Image.open(mask_rgb).convert("RGB"))

            mask = self.rgb_to_class(mask_rgb)

            if self.transform:
                augmented = self.transform(image=image, mask=mask)
                image = augmented["image"]
                mask = augmented["mask"]

            return image, mask
        except Exception as e:
            print(f"Error loading {self.images[index]}: {str(e)}")
            dummy_image = torch.zeros(3, 256, 256) if self.transform else np.zeros((256, 256, 3))
            dummy_mask = torch.zeros(256, 256).long() if self.transform else np.zeros((256, 256))
            return dummy_image, dummy_mask

    def get_path(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])
        mask_path = os.path.join(self.mask_dir, self.images[index].replace("_sat.jpg", "_mask.png"))
        return img_path, mask_path

    def rgb_to_class(self, mask_rgb):
        h, w = mask_rgb.shape[:2]
        mask_class = np.full((h, w), 6, dtype=np.int64)

        for rgb, class_idx in self.colour_mapping.items():
            # Find all pixels matching the current RGB value and assign the corresponding class index
            mask_class[(mask_rgb == rgb).all(axis=-1)] = class_idx

        return mask_class
    
    def class_distribution(self):
        """Returns pixel distribution across classes"""
        num_class = np.zeros(len(self.colour_mapping))
        for _, mask in self:
            if isinstance(mask, torch.Tensor):
                mask = mask.numpy()
            for class_idx in range(len(self.colour_mapping)):
                num_class[class_idx] += np.sum(mask == class_idx)
        return {self.label_mapping[i]: count for i, count in enumerate(num_class)}
        
    def summarize(self):
        print(f"Dataset Summary:")
        print(f"- Total samples: {len(self)}")
        print(f"- Image directory: {self.image_dir}")
        print(f"- Mask directory: {self.mask_dir}")
        print("\nClass Distribution:")
        dist = self.class_distribution()
        for cls, count in dist.items():
            print(f"{cls}: {count:,} pixels")

    def display_sample(self, index, figsize=(15, 7)):
        """
        Display image and mask with titles showing:
        - Filename
        - Image dimensions
        - Unique classes present in mask
        """
        # Get the sample
        img_filename = self.images[index]
        image, mask = self[index]

        mean = self.mean
        std = self.std
        
        # Convert tensors to numpy if needed
        if torch.is_tensor(image):
            image = image.numpy().transpose(1, 2, 0)  # CHW to HWC
            image = (image * np.array(std) + np.array(mean)).clip(0, 1)  # Un-normalize
        
        if torch.is_tensor(mask):
            mask = mask.numpy()
        
        # Create colored mask
        colored_mask = np.zeros((*mask.shape, 3), dtype=np.uint8)
        for color, class_idx in self.colour_mapping.items():
            colored_mask[mask == class_idx] = color
        
        # Get unique classes in mask
        unique_classes = np.unique(mask)
        class_names = self.label_mapping
        present_classes = [class_names[c] for c in unique_classes]
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Display image
        ax1.imshow(image)
        ax1.set_title(
            f"Image: {img_filename}\n"
            f"Shape: {image.shape}\n"
            f"Normalized: mean={mean}, std={std}",
            fontsize=10
        )
        ax1.axis('off')
        
        # Display mask
        ax2.imshow(colored_mask)
        ax2.set_title(
            f"Mask: {img_filename.replace('_sat.jpg', '_mask.png')}\n"
            f"Classes present: {', '.join(present_classes)}\n"
            f"Shape: {mask.shape}",
            fontsize=10
        )
        ax2.axis('off')
        
        # Add color legend
        legend_elements = [
            plt.Rectangle((0,0),1,1, color=np.array(color)/255, label=f'{class_idx}: {class_names[class_idx]}')
            for color, class_idx in self.colour_mapping.items()
        ]
        plt.legend(
            handles=legend_elements,
            bbox_to_anchor=(1.05, 1),
            loc='upper left',
            title="Class Colors"
        )
        
        plt.tight_layout()
        plt.show()