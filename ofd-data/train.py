import joblib
import re
import argparse
import pandas as pd
import sklearn
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder

mod1 = joblib.load('./models/first_model.sav')
mod2 = joblib.load('./models/second_model.sav')
mod3 = joblib.load('./models/third_model.sav')
le_1 = joblib.load('./dataPrep/labelEncoder1.sav')
le_2 = joblib.load('./dataPrep/labelEncoder2.sav')
tfidf_transformer = joblib.load('./dataPrep/tfidf_transformer.sav')
count_vect = joblib.load('./dataPrep/count_vect.sav')


def prepare(dataset):
    special_char_list = [':', ';', '?', '}', ')', '{', '(', '.', '/', '\/', ',', '"',
                         '-', '*']
    dataset['name_new'] = dataset['name'].apply(lambda x: x.lower())
    for special_char in special_char_list:
        dataset['name_new'] = dataset['name_new'].apply(lambda x: x.replace(special_char, ''))

    dataset['name_new'] = dataset['name_new'].apply(lambda x: ' '.join(re.findall('.....', x)))
    X_test_counts = count_vect.transform(dataset.name_new)
    X_test_tfidf = tfidf_transformer.transform(X_test_counts)

    return X_test_tfidf


def predictions(X_test_tfidf):
    predictionss = []
    predicts = mod1.predict(X_test_tfidf)
    for i, pr in enumerate(predicts):
        if pr == 1:
            pred = le_1.inverse_transform(mod2.predict(X_test_tfidf[i]))[0]
            if pred == 'DEMI CLICK':
                pred = mod3.predict(X_test_tfidf[i])
                predictionss.append(*le_2.inverse_transform(pred))
            else:
                predictionss.append(pred)
        else:
            predictionss.append('НЕ Сигареты ROTHMANS')
    return predictionss


parser = argparse.ArgumentParser()
parser.add_argument("filename", type=str,
                    help="filename")
args = parser.parse_args()

data = pd.read_csv(args.filename, sep='\t', encoding="cp1251",
                   engine='python')
datatfidf = prepare(data)
data['predict'] = predictions(datatfidf)
data.drop('name_new', axis=1, inplace=True)
data.to_csv('./predicted.txt', sep='\t')
