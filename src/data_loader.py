import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


def load_kaggle_data(news_path, prices_path):
    news_df = pd.read_csv(news_path)
    prices_df = pd.read_csv(prices_path)
    print(f"Loaded news: {news_df.shape}, prices: {prices_df.shape}")
    return news_df, prices_df


def create_headline_examples(news_df):
    headlines = []
    labels = []
    for idx, row in news_df.iterrows():
        day_label = row['Label']
        for i in range(1, 26):
            col_name = f'Top{i}'
            if col_name in row and pd.notna(row[col_name]):
                headline = str(row[col_name]).strip()
                if len(headline) > 0:
                    headlines.append(headline)
                    labels.append(day_label)
    labels = np.array(labels)
    print(f"Created {len(headlines)} examples. Balance: {np.mean(labels):.3f} UP")
    return headlines, labels


def split_data(X, y, val_size=0.15, test_size=0.15, random_state=42):
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=(val_size + test_size), random_state=random_state, stratify=y
    )
    split_ratio = val_size / (val_size + test_size)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=(1 - split_ratio), random_state=random_state, stratify=y_temp
    )
    print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    return X_train, X_val, X_test, y_train, y_val, y_test


def tokenize_and_pad(headlines_train, headlines_val, headlines_test,
                     vocab_size=5000, max_length=500):
    tokenizer = Tokenizer(num_words=vocab_size, oov_token='<OOV>')
    tokenizer.fit_on_texts(headlines_train)
    
    X_train_seq = tokenizer.texts_to_sequences(headlines_train)
    X_val_seq = tokenizer.texts_to_sequences(headlines_val)
    X_test_seq = tokenizer.texts_to_sequences(headlines_test)
    
    X_train_padded = pad_sequences(X_train_seq, maxlen=max_length, padding='post')
    X_val_padded = pad_sequences(X_val_seq, maxlen=max_length, padding='post')
    X_test_padded = pad_sequences(X_test_seq, maxlen=max_length, padding='post')
    
    print(f"Tokenized: vocab={len(tokenizer.word_index)}, shapes={X_train_padded.shape}")
    return X_train_padded, X_val_padded, X_test_padded, tokenizer


def get_headline_stats(headlines):
    lengths = [len(h.split()) for h in headlines]
    print(f"Headlines: {len(headlines)}, avg words: {np.mean(lengths):.1f}, max: {np.max(lengths)}")