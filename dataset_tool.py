import pandas as pd
import re
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import os


def load_data(train_path, label_path, token_path= 'tokens.csv',test_size=0.33):
    y_df = pd.read_csv(label_path)
    if not os.path.isfile(token_path):
        print("Pre-procesing data")
        x_df = pd.read_csv(train_path)
        spacy_nlp = spacy.load("fr")
        tqdm.pandas()
        x_df['tokens'] = x_df['designation'].progress_apply(lambda s: raw_to_tokens(s, spacy_nlp))
        x_df.to_csv('./tokens.csv')
    else:
        x_df = pd.read_csv('tokens.csv')
    x_df, y_df = x_df[~ x_df.tokens.isna()], y_df[~ x_df.tokens.isna()]
    tfidf = TfidfVectorizer()
    X = tfidf.fit_transform(x_df['tokens'])
    y = y_df.prdtypecode
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    return X_train, X_test, y_train, y_test


def normalize_accent(string):
    string = string.replace('á', 'a')
    string = string.replace('â', 'a')
    string = string.replace('é', 'e')
    string = string.replace('è', 'e')
    string = string.replace('ê', 'e')
    string = string.replace('ë', 'e')
    string = string.replace('î', 'i')
    string = string.replace('ï', 'i')
    string = string.replace('ö', 'o')
    string = string.replace('ô', 'o')
    string = string.replace('ò', 'o')
    string = string.replace('ó', 'o')
    string = string.replace('ù', 'u')
    string = string.replace('û', 'u')
    string = string.replace('ü', 'u')
    string = string.replace('ç', 'c')
    string = re.sub(r'°|#|/|-|%|_|[0-9]+', '', string)
    return string


def raw_to_tokens(raw_string, spacy_nlp):
    string = raw_string.lower().rstrip()
    string = normalize_accent(string)
    spacy_tokens = spacy_nlp(string)
    string_tokens = [token.orth_ for token in spacy_tokens if not token.is_punct if not token.is_stop]
    clean_string = " ".join(string_tokens)
    return clean_string
