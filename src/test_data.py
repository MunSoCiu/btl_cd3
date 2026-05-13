from src.load_data import load_data
from src.data_check import check_data
from src.preprocess import preprocess
from src.build_dataset import build_dataset

users, books, history = load_data()

check_data(users, books, history)

users, books, history = preprocess(users, books, history)

df = build_dataset(users, books, history)

print("\n✅ DATA READY:", df.shape)