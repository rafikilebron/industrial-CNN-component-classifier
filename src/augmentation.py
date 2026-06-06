import cv2
import random
import os
import shutil

input_folder    = r'C:\Users\RGG9TL\PycharmProjects\industrial-cnn-classifier\datasets\welding-check\train_images'
output_folder   = r'C:\Users\RGG9TL\PycharmProjects\industrial-cnn-classifier\datasets\welding-check\aug'

def random_brightness(img):
    # Brightness shift
    beta = random.randint(-30, 30)
    return cv2.convertScaleAbs(img, beta=beta)

def random_contrast(img):
    # Contrast 0.8 – 1.2 pass
    alpha = 0.8 + 0.4 * random.random()
    return cv2.convertScaleAbs(img, alpha=alpha)

def random_rotation(img, angle):
    angle = random.uniform(-angle, angle)
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1)
    return cv2.warpAffine(img, M, (w, h))

def random_vertical_flip(img):
    return cv2.flip(img, 0)

def random_horizontal_flip(img):
    return cv2.flip(img, 1)

def augment_image(img):
    option = random.randint(3, 4)

    #if      option == 1:        return random_brightness(img)
    #if      option == 2:        return random_contrast(img)
    if      option == 3:        return random_rotation(img, angle=2)
    #if      option == 4:        return random_vertical_flip(img)
    #if      option == 5:        return random_horizontal_flip(img)
    else:                       return img

########################################################################################################################

classes = ["OK", "NOK"]
n_augmentations = 100

for class_ in classes:
    print("-"*30)
    print(f"Augmentating '{class_}' images...")

    # Set folder path
    src_class = os.path.join(input_folder, class_)
    dst_class = os.path.join(output_folder, class_)

    # Create destination folder
    os.makedirs(dst_class, exist_ok=True)

    images = [f for f in sorted(os.listdir(src_class)) if f.endswith(('.jpg', '.jpeg', '.png', '.bmp'))]

    for i in range(n_augmentations):
        print(f"{i+1}/{n_augmentations} augmentation...")

        # Choose random image
        file = random.choice(images)

        src = os.path.join(src_class, file)

        # Augment
        aug = augment_image(cv2.imread(src))

        # UUID
        filename = f"{os.path.splitext(file)[0]}_aug_{i}.jpg"
        dst = os.path.join(dst_class, filename)

        cv2.imwrite(dst, aug)









