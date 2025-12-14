import os
import pandas as pd
from src.train import load_data, train_model

def test_data_loading():
    df = load_data("data/dataset.csv")
    assert not df.empty

def test_model_training():
    df = load_data("data/dataset.csv")
    model = train_model(df)
    assert model is not None

def test_model_saved():
    assert os.path.exists("models/model.pkl")
