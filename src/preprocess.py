import pandas as pd
from datetime import datetime
import random

def preprocess(users, books, history):

    # ===== NORMALIZE COLUMN =====
    users.columns = users.columns.str.strip().str.lower()
    books.columns = books.columns.str.strip().str.lower()
    history.columns = history.columns.str.strip().str.lower()

    # ===== RENAME STANDARD =====
    if 'user_id' in users.columns:
        users.rename(columns={'ser_id': 'user_id'}, inplace=True)

    if 'userid' in users.columns:
        users.rename(columns={'userid': 'user_id'}, inplace=True)

    if 'userid' in history.columns:
        history.rename(columns={'userid': 'user_id'}, inplace=True)

    if 'bookid' in history.columns:
        history.rename(columns={'bookid': 'book_id'}, inplace=True)

    # ===== VALIDATE =====
    if 'user_id' not in users.columns:
        raise Exception(f"❌ users thiếu user_id | columns: {users.columns.tolist()}")

    if 'user_id' not in history.columns:
        raise Exception(f"❌ history thiếu user_id | columns: {history.columns.tolist()}")

    if 'book_id' not in history.columns:
        raise Exception(f"❌ history thiếu book_id | columns: {history.columns.tolist()}")

    # ===== USERS =====
    users['ngay_sinh'] = pd.to_datetime(users['ngay_sinh'], errors='coerce')

    current_year = datetime.now().year
    users['tuoi'] = current_year - users['ngay_sinh'].dt.year

    users['age_group'] = pd.cut(
        users['tuoi'],
        bins=[0, 18, 25, 35, 50],
        labels=["<18", "18-25", "25-35", "35+"]
    )

    # ===== HISTORY =====
    history['ngay_bat_dau'] = pd.to_datetime(
        history['ngay_bat_dau'], errors='coerce'
    )

    history = history.dropna(subset=['ngay_bat_dau'])

    # ===== BOOK COUNT =====
    book_count = (
        history.groupby('user_id')['book_id']
        .nunique()
        .reset_index(name='so_luong_sach_da_mua')
    )

    users = users.merge(book_count, on='user_id', how='left')

       # ===== SAFE HANDLE COLUMN =====
    if 'so_luong_sach_da_mua' in users.columns:
        users['so_luong_sach_da_mua'] = users['so_luong_sach_da_mua'].fillna(0)
    else:
        print("⚠️ Missing column -> generating")

        books_count = history.groupby('user_id')['book_id'].count().reset_index()
        books_count.columns = ['user_id', 'so_luong_sach_da_mua']

        users = users.merge(books_count, on='user_id', how='left')
        users['so_luong_sach_da_mua'] = users['so_luong_sach_da_mua'].fillna(0)

    return users, books, history

def augment_users_realistic(users, books, history, n=80):
   
    # ===== FIX COLUMN =====
    books.columns = books.columns.str.strip()

    if 'the_loai' not in books.columns:
        raise Exception(f"❌ Không có cột 'the_loai'. Columns: {books.columns.tolist()}")

    current_year = datetime.now().year
    max_user_id = users['user_id'].max()

    profiles = {
        "<18": {
            "age_range": (10, 17),
            "genres": ["Manga", "Fantasy"],
            "rating": (4, 5),
            "drop_rate": 0.1,
            "books": 6
        },
        "18-25": {
            "age_range": (18, 25),
            "genres": ["Manga", "Romance", "Fantasy"],
            "rating": (3, 5),
            "drop_rate": 0.2,
            "books": 8
        },
        "25-35": {
            "age_range": (26, 35),
            "genres": ["Kỹ năng sống", "Kinh doanh"],
            "rating": (3, 5),
            "drop_rate": 0.15,
            "books": 7
        },
        "35+": {
            "age_range": (36, 60),
            "genres": ["Kỹ năng sống", "Lịch sử"],
            "rating": (3, 5),
            "drop_rate": 0.1,
            "books": 5
        }
    }

    new_users = []
    new_history = []

    # ===== GROUP BOOKS =====
    genre_books = books.groupby('the_loai')['book_id'].apply(list).to_dict()

    for i in range(n):
        group = random.choice(list(profiles.keys()))
        profile = profiles[group]

        age = random.randint(*profile["age_range"])
        birth_year = current_year - age

        user_id = max_user_id + i + 1

        new_users.append({
            "user_id": user_id,
            "ngay_sinh": f"{birth_year}-01-01"
        })

        for _ in range(profile["books"]):
            genre = random.choice(profile["genres"])

            if genre not in genre_books:
                genre = random.choice(list(genre_books.keys()))

            book_id = random.choice(genre_books[genre])
            
            rating = random.choices(
                [1,2,3,4,5],
                weights=[5,10,20,30,35]
            )[0]
            
            status = random.choices(
                ["done", "reading", "dropped"],
                weights=[60,25,15]
            )[0]

            new_history.append({
                "user_id": user_id,
                "book_id": book_id,
                "ngay_bat_dau": f"2024-{random.randint(1,12):02d}-{random.randint(1,28):02d}",
                "diem_danh_gia": rating,
                "trang_thai": status
            })

    users_new = pd.DataFrame(new_users)
    history_new = pd.DataFrame(new_history)

    users = pd.concat([users, users_new], ignore_index=True)
    history = pd.concat([history, history_new], ignore_index=True)

    print("🔥 Synthetic users added:", len(users_new))

    return users, history

def augment_books(books, n=100):
    max_id = books['book_id'].max()
    genres = books['the_loai'].unique()

    new_books = []

    for i in range(n):
        new_books.append({
            "book_id": max_id + i + 1,
            "ten_sach": f"Sách AI {i}",       
            "tac_gia": "AI Generated",
            "the_loai": random.choice(genres),  
            "so_trang": random.randint(100, 800),
            "loai_sach": random.choice(["Truyện chữ", "Truyện tranh"]),
            "diem_danh_gia": round(random.uniform(3,5),2),
            "nam_xuat_ban": random.randint(2010, 2024),
            "quoc_gia": "AI",
            "gia_sach": random.randint(80000, 350000),
            "price_range": "Medium",
            "rating_level": "Good"
        })

    books = pd.concat([books, pd.DataFrame(new_books)], ignore_index=True)

    print("📚 Synthetic books added:", n)

    return books