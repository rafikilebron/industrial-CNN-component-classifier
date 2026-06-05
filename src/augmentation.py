import cv2
import random
import os
import shutil

input_folder = r'C:\Users\RGG9TL\Pictures\FED-case'
output_folder = r'C:\Users\RGG9TL\Pictures\FED-aug'

def random_brightness(img):
    # Brightness shift
    beta = random.randint(-30, 30)
    return cv2.convertScaleAbs(img, beta=beta)

def random_contrast(img):
    # Contrast 0.8 – 1.2 pass
    alpha = 0.8 + 0.4 * random.random()
    return cv2.convertScaleAbs(img, alpha=alpha)

def random_rotation(img):
    angle = random.uniform(-5, 5)
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1)
    return cv2.warpAffine(img, M, (w, h))

def augment_image(img):
    option = random.randint(1, 3)

    if option == 1:
        return random_brightness(img)
    elif option == 2:
        return random_contrast(img)
    else:
        return random_rotation(img)


models = ['8VU', '255', '305']
n_augmentations = 100

for model in models:
    print("-"*30)
    print(f"Augmentating '{model}' images...")

    # Set folder path
    src_model = os.path.join(input_folder, model)
    dst_model = os.path.join(output_folder, model)

    # Create destination folder
    os.makedirs(dst_model, exist_ok=True)

    images = [f for f in sorted(os.listdir(src_model)) if f.endswith(('.jpg', '.jpeg', '.png', '.bmp'))]

    for i in range(n_augmentations):
        print(f"{i+1}/{n_augmentations} augmentation...")

        # Choose random image
        file = random.choice(images)

        src = os.path.join(src_model, file)

        # Augment
        aug = augment_image(cv2.imread(src))

        # UUID
        filename = f"{os.path.splitext(file)[0]}_aug_{i}.jpg"
        dst = os.path.join(dst_model, filename)

        cv2.imwrite(dst, aug)









