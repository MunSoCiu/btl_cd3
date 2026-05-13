import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data")

def load_data():
    try:
        users = pd.read_csv(os.path.join(DATA_PATH, "users.csv"))
        books = pd.read_csv(os.path.join(DATA_PATH, "books.csv"))
        history = pd.read_csv(os.path.join(DATA_PATH, "cleaned_reading_history.csv"))

        print("✅ Load data thành công")
        print(f"Users: {users.shape}")
        print(f"Books: {books.shape}")
        print(f"History: {history.shape}")

        return users, books, history

    except Exception as e:
        print("❌ Lỗi load data:", e)
        return None, None, None