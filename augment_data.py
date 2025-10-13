import os
import numpy as np
from PIL import Image
import random
from pathlib import Path
from tqdm import tqdm

class MedicalImageAugmentor:
    def __init__(self, input_dir, output_dir, augmentations_per_image=5):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.augmentations_per_image = augmentations_per_image
        
    def rotate_image(self, img, angle):
        return img.rotate(angle, resample=Image.BICUBIC, fillcolor=0)
    
    def flip_horizontal(self, img):
        return img.transpose(Image.FLIP_LEFT_RIGHT)
    
    def flip_vertical(self, img):
        return img.transpose(Image.FLIP_TOP_BOTTOM)
    
    def adjust_brightness(self, img, factor):
        img_array = np.array(img).astype(np.float32)
        img_array = np.clip(img_array * factor, 0, 255)
        return Image.fromarray(img_array.astype(np.uint8))
    
    def adjust_contrast(self, img, factor):
        img_array = np.array(img).astype(np.float32)
        mean = img_array.mean()
        img_array = mean + factor * (img_array - mean)
        img_array = np.clip(img_array, 0, 255)
        return Image.fromarray(img_array.astype(np.uint8))
    
    def add_gaussian_noise(self, img, std=5):
        img_array = np.array(img).astype(np.float32)
        noise = np.random.normal(0, std, img_array.shape)
        img_array = np.clip(img_array + noise, 0, 255)
        return Image.fromarray(img_array.astype(np.uint8))
    
    def zoom(self, img, zoom_factor):
        width, height = img.size
        new_width = int(width * zoom_factor)
        new_height = int(height * zoom_factor)
        img_resized = img.resize((new_width, new_height), Image.BICUBIC)
        left = (new_width - width) // 2
        top = (new_height - height) // 2
        right = left + width
        bottom = top + height
        if zoom_factor > 1:
            return img_resized.crop((left, top, right, bottom))
        else:
            new_img = Image.new(img.mode, (width, height), 0)
            new_img.paste(img_resized, ((width - new_width) // 2, (height - new_height) // 2))
            return new_img
    
    def translate(self, img, x_shift, y_shift):
        return img.transform(img.size, Image.AFFINE, (1, 0, x_shift, 0, 1, y_shift), fillcolor=0)
    
    def augment_single_image(self, img, aug_type, is_mask=False):
        if aug_type == 'rotate_small':
            angle = random.uniform(-15, 15)
            return self.rotate_image(img, angle)
        elif aug_type == 'rotate_medium':
            angle = random.choice([-30, -25, -20, 20, 25, 30])
            return self.rotate_image(img, angle)
        elif aug_type == 'flip_h':
            return self.flip_horizontal(img)
        elif aug_type == 'flip_v':
            return self.flip_vertical(img)
        elif aug_type == 'brightness' and not is_mask:
            factor = random.uniform(0.8, 1.2)
            return self.adjust_brightness(img, factor)
        elif aug_type == 'contrast' and not is_mask:
            factor = random.uniform(0.8, 1.2)
            return self.adjust_contrast(img, factor)
        elif aug_type == 'noise' and not is_mask:
            std = random.uniform(3, 8)
            return self.add_gaussian_noise(img, std)
        elif aug_type == 'zoom_in':
            factor = random.uniform(1.05, 1.15)
            return self.zoom(img, factor)
        elif aug_type == 'zoom_out':
            factor = random.uniform(0.85, 0.95)
            return self.zoom(img, factor)
        elif aug_type == 'translate':
            x_shift = random.randint(-20, 20)
            y_shift = random.randint(-20, 20)
            return self.translate(img, x_shift, y_shift)
        elif aug_type == 'combined' and not is_mask:
            img = self.rotate_image(img, random.uniform(-10, 10))
            img = self.adjust_brightness(img, random.uniform(0.9, 1.1))
            img = self.add_gaussian_noise(img, random.uniform(2, 5))
            return img
        elif aug_type == 'combined' and is_mask:
            img = self.rotate_image(img, random.uniform(-10, 10))
            return img
        return img
    
    def process_dataset(self):
        geometric_augs = ['rotate_small', 'rotate_medium', 'flip_h', 'flip_v', 'zoom_in', 'zoom_out', 'translate']
        intensity_augs = ['brightness', 'contrast', 'noise', 'combined']
        for split in ['train', 'val']:
            should_augment = (split == 'train')
            augmentation_plan = {}
            for data_type in ['img', 'gt']:
                input_path = self.input_dir / split / data_type
                output_path = self.output_dir / split / data_type
                if not input_path.exists():
                    print(f"Warning: {input_path} does not exist, skipping...")
                    continue
                output_path.mkdir(parents=True, exist_ok=True)
                image_files = sorted(list(input_path.glob('*.png')) + list(input_path.glob('*.jpg')))
                is_mask = (data_type == 'gt')
                print(f"\nProcessing {split}/{data_type}: {len(image_files)} images")
                for img_file in tqdm(image_files, desc=f"{split}/{data_type}"):
                    img = Image.open(img_file)
                    original_name = img_file.stem
                    img.save(output_path / f"{original_name}_original.png")
                    if not should_augment:
                        continue
                    if data_type == 'img':
                        geom_augs = random.sample(geometric_augs, min(self.augmentations_per_image, len(geometric_augs)))
                        total_augs = geom_augs.copy()
                        if len(total_augs) < self.augmentations_per_image:
                            remaining = self.augmentations_per_image - len(total_augs)
                            total_augs.extend(random.sample(intensity_augs, min(remaining, len(intensity_augs))))
                        augmentation_plan[original_name] = total_augs
                    selected_augs = augmentation_plan[original_name]
                    for i, aug_type in enumerate(selected_augs):
                        aug_img = self.augment_single_image(img, aug_type, is_mask=is_mask)
                        aug_filename = f"{original_name}_aug_{i+1}_{aug_type}.png"
                        aug_img.save(output_path / aug_filename)
                if should_augment:
                    print(f"Completed {split}/{data_type}: {len(image_files) * (self.augmentations_per_image + 1)} total images")
                else:
                    print(f"Completed {split}/{data_type}: {len(image_files)} original images (no augmentation)")

def main():
    INPUT_DIR = "SEGTHOR_CLEAN"
    OUTPUT_DIR = "SEGTHOR_AUGMENTED"
    AUGMENTATIONS_PER_IMAGE = 5
    augmentor = MedicalImageAugmentor(
        input_dir=INPUT_DIR,
        output_dir=OUTPUT_DIR,
        augmentations_per_image=AUGMENTATIONS_PER_IMAGE
    )
    print("Starting data augmentation...")
    print(f"Input: {INPUT_DIR}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Augmentations per image: {AUGMENTATIONS_PER_IMAGE}")
    print("-" * 50)
    augmentor.process_dataset()
    print("\n" + "=" * 50)
    print("Data augmentation completed successfully!")
    print(f"Augmented data saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
