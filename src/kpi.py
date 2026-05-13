# src/kpi.py

def compute_kpi(df):
    return {
        "total_users": df['user_id'].nunique(),
        "total_books": df['book_id'].nunique(),
        "avg_rating": df['rating_book'].mean(),
        "total_transactions": len(df)
    }