import os

OUTPUT = "output/data"

def save_data(df):
    import os
    os.makedirs("output/data", exist_ok=True)

    df.to_csv("output/data/cleaned_dataset.csv", index=False)

    print("✅ Data saved")

    # ===== UNIQUE BOOKS =====
    books_unique = df.drop_duplicates('book_id')

    # thống kê
    summary = books_unique.describe(include='all')
    summary.to_csv(f"{OUTPUT}/summary_stats.csv")

    # genre stats
    genre = books_unique['the_loai'].value_counts()
    genre.to_csv(f"{OUTPUT}/genre_stats.csv")


    print("✅ Data saved")