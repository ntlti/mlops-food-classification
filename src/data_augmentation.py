import os
import cv2
import shutil
import albumentations as A
from tqdm import tqdm

def augment_dataset(input_base_dir, output_base_dir, num_augment=3):
    # Очищаем и создаём выходную папку
    if os.path.exists(output_base_dir):
        shutil.rmtree(output_base_dir)
    os.makedirs(output_base_dir)

    transform = A.Compose([
        A.RandomRotate90(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=30, p=0.7),
        A.GaussNoise(p=0.3),
    ])

    for split in ['training', 'validation']:  # evaluation обычно не аугментируем
        input_dir = os.path.join(input_base_dir, split)
        output_dir = os.path.join(output_base_dir, split)

        if not os.path.exists(input_dir):
            print(f"Пропуск {split}: папка не найдена")
            continue

        os.makedirs(output_dir, exist_ok=True)

        for category in ['food', 'non_food']:
            cat_input = os.path.join(input_dir, category)
            cat_output = os.path.join(output_dir, category)
            os.makedirs(cat_output, exist_ok=True)

            if not os.path.exists(cat_input):
                continue

            images = [f for f in os.listdir(cat_input) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

            for img_name in tqdm(images, desc=f"{split}/{category}"):
                img_path = os.path.join(cat_input, img_name)
                image = cv2.imread(img_path)
                if image is None:
                    continue
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                # Сохраняем оригинал
                orig_path = os.path.join(cat_output, img_name)
                cv2.imwrite(orig_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

                # Генерируем аугментированные версии
                for i in range(num_augment):
                    augmented = transform(image=image)['image']
                    aug_name = f"aug_{i}_{img_name}"
                    aug_path = os.path.join(cat_output, aug_name)
                    cv2.imwrite(aug_path, cv2.cvtColor(augmented, cv2.COLOR_RGB2BGR))

    print("Аугментация завершена!")

if __name__ == "__main__":
    augment_dataset('data/original', 'data/augmented', num_augment=3)