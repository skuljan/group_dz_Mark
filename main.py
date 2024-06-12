import pandas as pd
import openai
import os
from sklearn.metrics import accuracy_score, classification_report

# Получение API ключа из переменной окружения
openai.api_key = os.getenv('OPENAI_API_KEY')

# Загрузка файла
file_path = 'sql_20240612.xlsx'
df = pd.read_excel(file_path, engine='openpyxl')

# Проверка структуры DataFrame
print("Колонки в DataFrame:", df.columns)

# Извлечение первых 100 отзывов из колонки message
if 'message' in df.columns:
    reviews = df['message'].head(2000).tolist()
else:
    print("Колонка 'message' не найдена в DataFrame")
    reviews = []

# Категории для анализа
categories = [
    "Органолептика", "Качество продукта", "Соотношение цена/качество",
    "Упаковка", "Доставка"
]

# Пояснения для категорий
category_explanations = """
Категории и их пояснения:
1) Если речь идет о вкусовых характеристиках, характеристиках текстуры, запаха, каких-то чувственных качествах продукта, то это органолептика.
2) Если речь идет о качестве именно продукта, без упоминания запаха, цвета или вкуса, то это качество продукта.
3) Если речь идет о том, что что-то слишком дешевое или слишком дорогое, то это соотношение цена/качество.
4) Если речь идет о характеристиках упаковки, то это категория Упаковка.
5) Если какие-то проблемы с доставкой или наоборот что-то хорошее с доставкой, то это категория Доставка.
"""

# Функция для анализа отзывов
def analyze_review(text):
    response = openai.ChatCompletion.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"{category_explanations}\n\nВыбери одну категорию для следующего отзыва из списка: {', '.join(categories)}. Отвечай только названием категории без объяснений и дополнительных слов. Отзыв: '{text}'"}
        ],
        max_tokens=20,
        temperature=0.0,  # Низкая температура для предсказуемых ответов
    )
    analysis = response['choices'][0]['message']['content'].strip()
    return analysis

# Анализ первых 100 отзывов, если колонка найдена
if reviews:
    categories_assigned = [analyze_review(review) for review in reviews]
    # Добавление полученных категорий в DataFrame
    df.loc[:1999, 'predicted_topic'] = categories_assigned

    # Сохранение результатов в новый файл
    output_file_path = 'review_analysis.xlsx'
    df.to_excel(output_file_path, index=False, engine='openpyxl')

    print("Анализ завершен. Результаты сохранены в файле review_analysis.xlsx")

    # Оценка точности модели на первых 50 отзывах
    if 'hand_topic' in df.columns:
        hand_labeled = df['hand_topic'].head(50).tolist()
        predicted = df['predicted_topic'].head(50).tolist()

        # Убедимся, что все категории присутствуют в целевых метках
        labels = categories

        accuracy = accuracy_score(hand_labeled, predicted)
        report = classification_report(hand_labeled, predicted, target_names=categories, labels=labels)

        print(f"Accuracy: {accuracy:.2f}")
        print("Classification Report:")
        print(report)

        # Сохранение отчета в текстовый файл
        with open("classification_report.txt", "w") as file:
            file.write(f"Accuracy: {accuracy:.2f}\n")
            file.write("Classification Report:\n")
            file.write(report)

        print("Отчет сохранен в файл classification_report.txt")
    else:
        print("Колонка 'hand_topic' не найдена в DataFrame")
else:
    print("Анализ не выполнен, так как колонка 'message' не найдена.")