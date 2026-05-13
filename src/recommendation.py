import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


# ===== USER SIMILARITY =====
def compute_user_similarity(history, books):
    df = history.merge(books, on="book_id", how="left")

    # ===== FIX RATING COLUMN =====
    if 'diem_danh_gia' not in df.columns:

       if 'diem_danh_gia_x' in df.columns:
           df['diem_danh_gia'] = df['diem_danh_gia_x']

       elif 'rating_book' in df.columns:
           df['diem_danh_gia'] = df['rating_book']

       else:
           raise Exception(f"❌ Missing rating column | columns: {df.columns.tolist()}")

    pivot = df.pivot_table(
        index="user_id",   
        columns="book_id",
        values="diem_danh_gia"
    )

    pivot = pivot.fillna(pivot.mean())

    from sklearn.metrics.pairwise import cosine_similarity
    sim = cosine_similarity(pivot)

    return pd.DataFrame(sim, index=pivot.index, columns=pivot.index)


# ===== USER PROFILE =====
def build_user_profile(history, books):
    df = history.merge(books, on="book_id", how="left")

    user_pref = df.groupby(['user_id', 'the_loai']).size().reset_index(name='count')

    user_top = user_pref.sort_values(['user_id', 'count'], ascending=False)
    user_top = user_top.groupby('user_id').first().reset_index()

    return user_top


# ===== POPULARITY =====
def compute_popularity(history):
    pop = history['book_id'].value_counts().reset_index()
    pop.columns = ['book_id', 'popularity']
    return pop


# ===== HYBRID RECOMMEND =====
def recommend_books_hybrid(user_id, books, history, user_top, sim_df, top_n=5):

    read_books = history[history['user_id'] == user_id]['book_id']

    fav = user_top[user_top['user_id'] == user_id]
    fav_genre = fav.iloc[0]['the_loai'] if not fav.empty else None

    pop = compute_popularity(history)

    df = books.drop_duplicates('book_id').merge(pop, on='book_id', how='left').fillna(0)

    # ===== SCORES =====
    df['genre_score'] = (df['the_loai'] == fav_genre).astype(int)

    df['rating_score'] = df['diem_danh_gia'].fillna(3) / 5

    df['pop_score'] = df['popularity'] / (df['popularity'].max() + 1)

    # ===== SIM SCORE =====
    sim_score = {}

    if user_id in sim_df.index:
        similar_users = sim_df[user_id].sort_values(ascending=False)[1:6]

        for u, s in similar_users.items():
            books_u = history[history['user_id'] == u]

            for _, row in books_u.iterrows():
                b = row['book_id']
                if b in read_books.values:
                    continue

                sim_score[b] = sim_score.get(b, 0) + s * row['diem_danh_gia']

    df['sim_score'] = df['book_id'].map(sim_score).fillna(0)

    # normalize sim
    if df['sim_score'].max() > 0:
        df['sim_score'] = df['sim_score'] / df['sim_score'].max()

    # ===== FINAL =====
    df['final_score'] = (
        0.4 * df['genre_score'] +
        0.3 * df['rating_score'] +
        0.2 * df['pop_score'] +
        0.1 * df['sim_score']
    )

    df = df[~df['book_id'].isin(read_books)]

    return df.sort_values('final_score', ascending=False).head(top_n)


# ===== GENERATE ALL =====
def generate_all_recommendations(users, books, history):
    user_top = build_user_profile(history, books)
    sim_df = compute_user_similarity(history, books)

    results = []

    for user_id in users['user_id']:
        recs = recommend_books_hybrid(
            user_id, books, history, user_top, sim_df
        )

        if not recs.empty:
            recs = recs.copy()
            recs['user_id'] = user_id
            results.append(recs)

    return pd.concat(results) if results else pd.DataFrame()