import os
import matplotlib.pyplot as plt
import seaborn as sns

OUTPUT_PATH = "output/charts"

def ensure_dir():
    os.makedirs(OUTPUT_PATH, exist_ok=True)

def save_plot(name):
    plt.savefig(f"{OUTPUT_PATH}/{name}.png", dpi=300, bbox_inches='tight')
    plt.close()

def plot_all(df, users, books, history):
    ensure_dir()

    # ===== UNIQUE BOOKS =====
    books_unique = books.drop_duplicates('book_id')

    # ===== 1. GENRE =====
    plt.figure()
    sns.countplot(data=books_unique, x="the_loai",
                  order=books_unique['the_loai'].value_counts().index)
    plt.xticks(rotation=30)
    plt.title("Genre Distribution")
    save_plot("genre")

    # ===== 2. RATING DISTRIBUTION =====
    plt.figure()
    sns.histplot(books_unique['diem_danh_gia'], kde=True)
    plt.title("Rating Distribution")
    save_plot("rating_dist")

    # ===== 3. AGE vs BOOKS =====
    user_books = history.groupby('user_id')['book_id'].nunique().reset_index()
    user_books.columns = ['user_id', 'so_luong_sach_da_mua']

    df_age = users.merge(user_books, on='user_id', how='left')
    df_age['so_luong_sach_da_mua'] = df_age['so_luong_sach_da_mua'].fillna(0)

    plt.figure()
    sns.barplot(data=df_age, x="age_group", y="so_luong_sach_da_mua")
    plt.title("Age vs Books")
    save_plot("age_books")

    # ===== 4. PRICE vs RATING =====
    plt.figure()
    sns.scatterplot(data=books_unique, x="gia_sach", y="diem_danh_gia")
    plt.title("Price vs Rating")
    save_plot("price_rating")

    # ===== 5. HEATMAP =====
    pivot = history.pivot_table(
        index="user_id",
        columns="book_id",
        values="diem_danh_gia",
        aggfunc='mean'
    ).fillna(0)

    plt.figure(figsize=(12,6))
    sns.heatmap(pivot, cmap="YlOrRd")
    plt.title("User-Book Heatmap")
    save_plot("heatmap")
    ensure_dir()

    # 1. Genre
    plt.figure()
    sns.countplot(data=books, x="the_loai")
    plt.xticks(rotation=30)
    plt.title("Genre Distribution")
    save_plot("genre")

    # 2. Rating Distribution
    plt.figure()
    sns.histplot(df['rating_book'], kde=True)
    plt.title("Rating Distribution")
    save_plot("rating_dist")

    # 👉 đảm bảo có cột
    if 'so_luong_sach_da_mua' not in df.columns:
        df['so_luong_sach_da_mua'] = df.groupby('user_id')['book_id'].transform('nunique')

    # 3. Age vs Books
    df_age = df[['user_id','age_group','so_luong_sach_da_mua']].drop_duplicates()

    plt.figure()
    sns.barplot(data=df_age, x="age_group", y="so_luong_sach_da_mua")
    plt.title("Age vs Books")
    save_plot("age_books")

    # 4. Price vs Rating
    plt.figure()
    sns.scatterplot(data=df, x="gia_sach", y="rating_book")
    plt.title("Price vs Rating")
    save_plot("price_rating")

    # 5. Heatmap
    pivot = history.pivot_table(
        index="user_id",
        columns="book_id",
        values="diem_danh_gia"
    ).fillna(0)

    plt.figure(figsize=(12,6))
    sns.heatmap(pivot, cmap="YlOrRd")
    plt.title("User-Book Heatmap")
    save_plot("heatmap")