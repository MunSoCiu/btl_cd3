import pandas as pd

sim = pd.read_csv("../output/data/user_similarity.csv", index_col=0)
df = pd.read_csv("../output/data/cleaned_dataset.csv")

def recommend(user_id, top_n=5):

    similar_users = sim[user_id].sort_values(ascending=False)[1:6]

    print("👤 Similar users:", similar_users.index.tolist())

    # sách user đã đọc
    read_books = df[df['user_id'] == user_id]['book_id']

    # sách từ user giống
    rec_books = df[df['user_id'].isin(similar_users.index)]

    rec_books = rec_books[~rec_books['book_id'].isin(read_books)]

    top_books = (
        rec_books.groupby('book_id')['rating_book']
        .mean()
        .sort_values(ascending=False)
        .head(top_n)
    )

    print("\n📚 Recommended books:", top_books.index.tolist())

recommend(1)