import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import load_model


MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)


def create_lstm_model(
    vectorizer,
    max_tokens,
    embedding_dim=64,
    lstm_units=64,
    bidirectional=False,
    dense_units=None,
    dropout_rate=0.0,
):
    model = Sequential()
    model.add(vectorizer)
    model.add(Embedding(input_dim=max_tokens, output_dim=embedding_dim, mask_zero=True))

    lstm_layer = LSTM(
        lstm_units,
        dropout=dropout_rate,
        recurrent_dropout=dropout_rate,
    )

    if bidirectional:
        model.add(Bidirectional(lstm_layer))
    else:
        model.add(lstm_layer)

    if dense_units is not None:
        model.add(Dense(dense_units, activation="relu"))
        if dropout_rate > 0:
            model.add(Dropout(dropout_rate))

    model.add(Dense(1, activation="sigmoid"))

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    return model

def train_model(
    model,
    train_ds,
    val_ds,
    epochs=10,
    patience=3,
    class_weight=None,
):
    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=patience,
        restore_best_weights=True,
    )

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=[early_stopping],
        class_weight=class_weight,
        verbose=1,
    )

    return history

def evaluate_model(model, dataset, y_true, threshold=0.5):
    probs = model.predict(dataset, verbose=0).ravel()
    preds = (probs >= threshold).astype(int)

    metrics = {
        "accuracy": accuracy_score(y_true, preds),
        "precision": precision_score(y_true, preds, zero_division=0),
        "recall": recall_score(y_true, preds, zero_division=0),
        "f1": f1_score(y_true, preds, zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, preds),
        "mean_probability": float(np.mean(probs)),
        "positive_rate": float(np.mean(preds)),
    }

    return metrics, probs, preds


def evaluate_constant_baseline(y_true, constant_prediction):
    preds = np.full(len(y_true), constant_prediction)

    metrics = {
        "accuracy": accuracy_score(y_true, preds),
        "precision": precision_score(y_true, preds, zero_division=0),
        "recall": recall_score(y_true, preds, zero_division=0),
        "f1": f1_score(y_true, preds, zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, preds),
        "mean_probability": float(constant_prediction),
        "positive_rate": float(np.mean(preds)),
    }

    return metrics, preds


def compare_metrics(results_dict):
    rows = []
    for name, m in results_dict.items():
        rows.append({
            "model": name,
            "accuracy": m["accuracy"],
            "precision": m["precision"],
            "recall": m["recall"],
            "f1": m["f1"],
        })

    return pd.DataFrame(rows).sort_values("accuracy", ascending=False)


def save_model(model, name):
    path = MODELS_DIR / f"{name}.keras"
    model.save(path)


def load_saved_model(name):
    path = MODELS_DIR / f"{name}.keras"
    return load_model(path)


def save_history(history, name):
    path = MODELS_DIR / f"{name}_history.json"
    with open(path, "w") as f:
        json.dump(history.history, f)


def save_vectorizer_vocab(vectorizer, name):
    path = MODELS_DIR / f"{name}_vocab.pkl"
    with open(path, "wb") as f:
        pickle.dump(vectorizer.get_vocabulary(), f)