import pandas as pd
from tqdm import tqdm
from transformers import pipeline
from sklearn.metrics import classification_report

# читаем файл
in_path = "sql_20240612.xlsx"
df = pd.read_excel(in_path)

# размечаем истинную тональность
df['ground_truth_label'] = 'positive'
df.loc[df['rating_value']<400, 'ground_truth_label'] = 'negative'

# подгружаем предобученную модель
model = pipeline(model="seara/rubert-tiny2-russian-sentiment")


def get_sentiment(text):
    """
    Функция для предсказания тональности, пишем ее, так как
    наша модель предсказывает 3 метки (положительный, отрицательный, нейтральный)
    и их вероятности, а наша задача бинарная (положительный, отрицательный)
    """
    scores = model(text, top_k=3)
    pos = [item['score'] for item in scores if item['label']=='positive'][0]
    neg = [item['score'] for item in scores if item['label']=='negative'][0]
    score = pos - neg
    return 'positive' if score >= 0 else 'negative'


# предсказываем тональность
labels_pred = []
for review in tqdm(df['message'], leave=False,
                   desc='Sentiment analysis'):
    labels_pred.append(get_sentiment(review))

df['predicted_label'] = labels_pred



# принтуем метрики
cr = classification_report(
    df['ground_truth_label'], df['predicted_label']
)
print(f'Classification report:\n{cr}')

# сохраняем новый файл с метками классов
out_path = "sentiment_analysis.xlsx"
df.to_excel(out_path)
print('Predictions saved to sentiment_analysis.xlsx')

# сохраняем файл с метриками
out_path = "sentiment_classification_report.txt"
with open(out_path, "w") as f:
    f.write(cr)
print('Classification report saved to sentiment_classification_report.txt')