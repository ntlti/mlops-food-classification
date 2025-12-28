import argparse
import mlflow
import mlflow.keras
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.applications import ResNet50, MobileNetV2
from sklearn.metrics import classification_report
from src.utils import create_data_generators

EPOCHS = 12
MODELS = ['simple_cnn', 'resnet50', 'mobilenetv2']

def build_model(model_type):
    if model_type == 'simple_cnn':
        model = Sequential([
            tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(224,224,3)),
            tf.keras.layers.MaxPooling2D(2,2),
            tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
            tf.keras.layers.MaxPooling2D(2,2),
            tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
            tf.keras.layers.MaxPooling2D(2,2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
    elif model_type == 'resnet50':
        base = ResNet50(weights='imagenet', include_top=False, input_shape=(224,224,3))
        base.trainable = False
        model = Sequential([
            base,
            Flatten(),
            Dense(256, activation='relu'),
            Dropout(0.5),
            Dense(1, activation='sigmoid')
        ])
    elif model_type == 'mobilenetv2':
        base = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224,224,3))
        base.trainable = False
        model = Sequential([
            base,
            Flatten(),
            Dense(256, activation='relu'),
            Dropout(0.5),
            Dense(1, activation='sigmoid')
        ])
    else:
        raise ValueError("Неизвестный тип модели")

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def run_experiments(data_dir, experiment_name):
    mlflow.set_experiment(experiment_name)

    train_gen, val_gen, test_gen = create_data_generators(data_dir)

    for model_type in MODELS:
        with mlflow.start_run(run_name=model_type):
            mlflow.log_param("model_type", model_type)
            mlflow.log_param("epochs", EPOCHS)
            mlflow.log_param("batch_size", train_gen.batch_size)
            mlflow.log_param("data_dir", data_dir)

            model = build_model(model_type)

            history = model.fit(
                train_gen,
                epochs=EPOCHS,
                validation_data=val_gen,
                verbose=1
            )

            # Логируем метрики
            final_val_acc = history.history['val_accuracy'][-1]
            final_val_loss = history.history['val_loss'][-1]
            mlflow.log_metric("val_accuracy", final_val_acc)
            mlflow.log_metric("val_loss", final_val_loss)

            # Оценка на тесте
            test_loss, test_acc = model.evaluate(test_gen, verbose=0)
            mlflow.log_metric("test_accuracy", test_acc)
            mlflow.log_metric("test_loss", test_loss)

            # Classification report
            preds = (model.predict(test_gen) > 0.5).astype(int)
            report = classification_report(test_gen.classes, preds, output_dict=True)
            mlflow.log_metric("precision", report['1']['precision'])
            mlflow.log_metric("recall", report['1']['recall'])
            mlflow.log_metric("f1_score", report['1']['f1-score'])

            report_text = classification_report(test_gen.classes, preds)
            mlflow.log_text(report_text, "classification_report.txt")

            # Сохраняем модель
            mlflow.keras.log_model(model, "model")

            print(f"{model_type} завершён: val_acc = {final_val_acc:.4f}, test_acc = {test_acc:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="data/original",
                        help="Путь к папке с данными: data/original или data/augmented")
    parser.add_argument("--experiment", type=str, default="FoodClassification",
                        help="Имя эксперимента в MLflow")
    args = parser.parse_args()

    exp_name = args.experiment + ("_Augmented" if "augmented" in args.data_dir else "_Original")
    print(f"Запуск экспериментов: {exp_name} на данных {args.data_dir}")
    run_experiments(args.data_dir, exp_name)