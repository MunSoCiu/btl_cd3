
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np

print("🤖 Training Recommendation Model...")

# ===== LOAD DATA =====
df = pd.read_csv("../output/data/cleaned_dataset.csv")

train, test = train_test_split(df, test_size=0.2, random_state=42)

# ===== USER-ITEM MATRIX =====
pivot = train.pivot_table(
    index="user_id",
    columns="book_id",
    values="rating_book"
)
pivot = pivot.fillna(pivot.mean())

# similarity
similarity = cosine_similarity(pivot)
sim_df = pd.DataFrame(similarity, index=pivot.index, columns=pivot.index)

# predict test
preds = []
actuals = []

for _, row in test.iterrows():
    user = row['user_id']
    book = row['book_id']

    if user in sim_df.index:
        similar_users = sim_df[user].sort_values(ascending=False)[1:6]

        weighted_sum = 0
        sim_sum = 0

        for sim_user, score in similar_users.items():
            user_data = train[train['user_id'] == sim_user]
            book_rating = user_data[user_data['book_id'] == book]['rating_book']

            if not book_rating.empty:
                weighted_sum += book_rating.values[0] * score
                sim_sum += score

        if sim_sum > 0:
            pred = weighted_sum / sim_sum
        else:
            pred = df['rating_book'].mean()
    else:
        pred = df['rating_book'].mean()

    preds.append(pred)
    actuals.append(row['rating_book'])

# ===== METRICS =====
rmse = np.sqrt(mean_squared_error(actuals, preds))
mae = np.mean(np.abs(np.array(actuals) - np.array(preds)))

print("RMSE:", rmse)
print("MAE:", mae)

# ===== SAVE MODEL =====
sim_df.to_csv("../output/data/user_similarity.csv")

print("✅ Model trained & saved")