# Deep learning
from torchvision.datasets import ImageFolder
from torchvision import transforms

# Data Processing & File management
import os
import shutil
import random


class DatasetCreator:
    def __init__(self, source_dir, output_dir, train_ratio=0.9):
        self.source_dir     = source_dir
        self.output_dir     = output_dir
        self.train_ratio    = train_ratio

    def create_dataset(self, train_transform, val_transform):
        print("Creating dataset...")

        try:
            # Create label -> images map
            self._list_label_to_images()

        except FileNotFoundError as e:
            print(e)
            print("Split might have been done already.")

        else:
            # Split images into train & validation
            self._split_dataset()

        finally:
            # Transform dataset
            train_dataset, val_dataset = self._transform_dataset(train_transform, val_transform)
            print("\nDataset successfully created")

        return train_dataset, val_dataset

    def _list_label_to_images(self):
        self.label_to_images = {}
        valid_ext = ('.jpg', '.jpeg', '.png', '.bmp')

        # Verify if source dir exists
        if not os.path.exists(self.source_dir):
            raise FileNotFoundError(f"Directory '{self.source_dir}' does not exist.")

        # Verify if source directory is not a file
        if not os.path.isdir(self.source_dir):
            raise NotADirectoryError(f"Directory '{self.source_dir}' is not a directory.")

        # Read directory
        try:
            # Loop over every class
            for label in os.listdir(self.source_dir):
                label_path = os.path.join(self.source_dir, label)

                # Ignore if not a folder
                if not os.path.isdir(label_path):
                    continue

                # Loop over every image
                for file_path in os.listdir(os.path.join(self.source_dir, label)):
                    # Verify if file is an image
                    if not file_path.endswith(valid_ext):
                        continue

                    # Store file name
                    self.label_to_images.setdefault(str(label), []).append(file_path)

        except PermissionError as e:
            raise PermissionError(f"Cannot access '{self.source_dir}'") from e

        if len(self.label_to_images) == 0:
            raise ValueError("No images found in '{self.source_dir}'")

    def _split_dataset(self):
        n_train = 0
        n_validation = 0

        print("Splitting dataset...")

        # Divide train/validation
        for label, image_list in self.label_to_images.items():

            # Shuffle the list of images
            random.shuffle(image_list)

            # Calculate the number of training images
            train_cutoff = int(len(image_list) * self.train_ratio)
            val_cutoff = int(len(image_list) - train_cutoff)
            n_train += train_cutoff
            n_validation += val_cutoff

            # Split images
            train_images = image_list[:train_cutoff]
            val_images = image_list[train_cutoff:]

            for split, split_images in zip(['train', 'val'], [train_images, val_images]):
                out_path = os.path.join(self.output_dir, split, label)
                os.makedirs(out_path, exist_ok=True)

                for img_name in split_images:
                    # Define paths
                    src_folder = os.path.join(self.source_dir, label)
                    src_image = os.path.join(src_folder, img_name)
                    dst = os.path.join(self.output_dir, split, label)

                    # Move image
                    shutil.move(src_image, dst)

        # Delete source folder
        shutil.rmtree(self.source_dir)

        print(f"Train/Val split complete -> {self.train_ratio*100:.2f}% training")
        print(f"{n_train} train samples\n{n_validation} validation samples")

    def _transform_dataset(self, train_transform, val_transform, gray=False):
        print("Transforming dataset...")

        # Include grayscale transformation
        if gray:
            train_transform.transforms.insert(1, transforms.Grayscale(num_output_channels=1))
            val_transform.transforms.insert(1, transforms.Grayscale(num_output_channels=1))

        train_dataset = ImageFolder(self.output_dir+'/train', transform=train_transform)
        val_dataset = ImageFolder(self.output_dir+'/val', transform=val_transform)

        return train_dataset, val_dataset