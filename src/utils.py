import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Глобальные параметры
IMG_SIZE = (224, 224)
BATCH_SIZE = 32


def create_data_generators(data_dir):
    """
    Создаёт генераторы для train, val, test.
    Test всегда берётся из data/original/evaluation для честной оценки.
    """
    # Пути к training и validation — из переданного data_dir (original или augmented)
    train_dir = os.path.join(data_dir, "training")
    val_dir = os.path.join(data_dir, "validation")

    # Надёжный расчёт корня проекта: поднимаемся на два уровня от utils.py
    # utils.py находится в src/, src/ в корне проекта → "..", ".."
    current_file_dir = os.path.dirname(os.path.abspath(__file__))        # → .../src
    project_root = os.path.abspath(os.path.join(current_file_dir, ".."))  # → корень проекта

    original_data_dir = os.path.join(project_root, "data", "original")
    test_dir = os.path.join(original_data_dir, "evaluation")

    # Проверка существования путей (для отладки)
    if not os.path.exists(train_dir):
        raise FileNotFoundError(f"Не найдена папка training: {train_dir}")
    if not os.path.exists(val_dir):
        raise FileNotFoundError(f"Не найдена папка validation: {val_dir}")
    if not os.path.exists(test_dir):
        raise FileNotFoundError(f"Не найдена папка evaluation в original: {test_dir}")

    # Data generators
    train_datagen = ImageDataGenerator(rescale=1. / 255)
    val_test_datagen = ImageDataGenerator(rescale=1. / 255)

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='binary',
        shuffle=True,
        seed=42
    )

    val_generator = val_test_datagen.flow_from_directory(
        val_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='binary',
        shuffle=False
    )

    test_generator = val_test_datagen.flow_from_directory(
        test_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='binary',
        shuffle=False
    )

    print(f"Train: {train_generator.samples} изображений (из {data_dir})")
    print(f"Val:   {val_generator.samples} изображений (из {data_dir})")
    print(f"Test:  {test_generator.samples} изображений (из original — честный тест)")

    return train_generator, val_generator, test_generator