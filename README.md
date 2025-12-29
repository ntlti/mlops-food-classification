# MLOps: Классификация Food vs Non-Food

Практическая работа по организации MLOps-процесса:

- Версионирование экспериментов с MLflow
- Аугментация данных
- Автоматический пайплайн обучения
- Сравнение моделей на исходных и аугментированных данных

## Как запустить

1. Скачайте датасет: https://www.kaggle.com/datasets/trolukovich/food5k-image-dataset
2. Распакуйте в `data/original/` (структура: training/, validation/, evaluation/ с папками food и non_food)
3. Установите зависимости:
   ```bash
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```
