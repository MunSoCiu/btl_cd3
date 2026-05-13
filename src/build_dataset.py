def build_dataset(users, books, history):

    df = history.merge(books, on="book_id", how="left")
    df = df.merge(users, on="user_id", how="left")

    df = df.rename(columns={
        "diem_danh_gia_x": "rating_user",
        "diem_danh_gia_y": "rating_book"
    })

    # ✅ đảm bảo cột tồn tại
    if 'so_luong_sach_da_mua' not in df.columns:
        df['so_luong_sach_da_mua'] = df.groupby('user_id')['book_id'].transform('nunique')

    return df