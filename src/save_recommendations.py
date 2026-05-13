import os
import pandas as pd
from src.recommendation import generate_all_recommendations

OUTPUT_PATH = "output/recommendations"


def save_recommendations(users, books, history):
    # ===== TẠO THƯ MỤC =====
    os.makedirs(OUTPUT_PATH, exist_ok=True)

    # ===== GENERATE =====
    result = generate_all_recommendations(users, books, history)

    if result.empty:
        print("⚠️ No recommendations generated")
        return

    # ===== CHUẨN HÓA BOOKS =====
    books = books.copy()
    books.columns = books.columns.str.strip().str.lower()

    rename_map = {}
    if 'title' in books.columns:
        rename_map['title'] = 'ten_sach'
    if 'genre' in books.columns:
        rename_map['genre'] = 'the_loai'

    books = books.rename(columns=rename_map)

    print("📚 BOOKS COLS:", books.columns.tolist())

    # ===== ĐẢM BẢO CỘT =====
    if 'ten_sach' not in books.columns:
        books['ten_sach'] = "Unknown"

    if 'the_loai' not in books.columns:
        books['the_loai'] = "Unknown"

    # ===== MERGE =====
    books_unique = books.drop_duplicates('book_id')

    result = result.merge(
        books_unique[['book_id', 'ten_sach', 'the_loai']],
        on='book_id',
        how='left'
    )

    print("📊 RESULT COLS:", result.columns.tolist())

    # ===== FIX COLUMN _x, _y =====
    if 'ten_sach_x' in result.columns:
        result['ten_sach'] = result['ten_sach_x']
    elif 'ten_sach_y' in result.columns:
        result['ten_sach'] = result['ten_sach_y']

    if 'the_loai_x' in result.columns:
        result['the_loai'] = result['the_loai_x']
    elif 'the_loai_y' in result.columns:
        result['the_loai'] = result['the_loai_y']

    # ===== FILL NA =====
    result['ten_sach'] = result['ten_sach'].fillna("Unknown")
    result['the_loai'] = result['the_loai'].fillna("Unknown")

    # ===== SELECT =====
    result = result[['user_id', 'book_id', 'ten_sach', 'the_loai', 'final_score']]

    # ===== SORT =====
    result = result.sort_values(['user_id', 'final_score'], ascending=[True, False])

    # ===== SAVE =====
    path = os.path.join(OUTPUT_PATH, "recommendations.csv")
    result.to_csv(path, index=False)

    print(f"✅ Saved recommendations → {path}")