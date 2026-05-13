import pandas as pd
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime

# ================= STYLE =================
sns.set_theme(style="whitegrid", palette="Set2")
plt.rcParams.update({
    "figure.figsize": (10,6),
    "axes.titlesize": 16,
    "axes.labelsize": 12
})

# ================= LOAD =================
users = pd.read_csv("users.csv")
books = pd.read_csv("books.csv")
history = pd.read_csv("cleaned_reading_history.csv")

print("✅ Load dữ liệu thành công")

# ================= CLEAN =================

# datetime
history['ngay_bat_dau'] = pd.to_datetime(history['ngay_bat_dau'], errors='coerce')
users['ngay_sinh'] = pd.to_datetime(users['ngay_sinh'], errors='coerce')

history = history.dropna(subset=['ngay_bat_dau'])

# tuổi
users['tuoi'] = datetime.now().year - users['ngay_sinh'].dt.year

users['age_group'] = pd.cut(
    users['tuoi'],
    bins=[0,18,25,35,50],
    labels=["<18","18-25","25-35","35+"]
)

# số sách đọc
book_count = history.groupby('user_id')['book_id'].nunique().reset_index()
book_count = book_count.rename(columns={'book_id':'so_luong_sach_da_mua'})

users = users.drop(columns=['so_luong_sach_da_mua'], errors='ignore')
users = users.merge(book_count, on='user_id', how='left')
users['so_luong_sach_da_mua'] = users['so_luong_sach_da_mua'].fillna(0)

# ================= MERGE =================
df = history.merge(books, on="book_id").merge(users, on="user_id")

df = df.rename(columns={
    "diem_danh_gia_x": "rating_user",
    "diem_danh_gia_y": "rating_book"
})

books_unique = df.drop_duplicates('book_id')

# ================= EDA =================

# Genre
plt.figure()
sns.countplot(data=books_unique, x="the_loai",
              order=books_unique['the_loai'].value_counts().index)
plt.xticks(rotation=30)
plt.title("So luong sach theo the loai")
plt.savefig("genre.png")
plt.close()

# Top book
top_books = df.groupby('book_id').agg({
    'ten_sach':'first',
    'rating_book':'mean'
}).sort_values(by='rating_book', ascending=False).head(10)

plt.figure()
sns.barplot(data=top_books, x="rating_book", y="ten_sach")
plt.title("Top 10 sach rating cao")
plt.savefig("top_rating.png")
plt.close()

# Status
plt.figure()
sns.countplot(data=history, x="trang_thai")
plt.title("Trang thai doc sach")
plt.savefig("status.png")
plt.close()

# User
plt.figure()
history['user_id'].value_counts().head(10).plot(kind='bar')
plt.title("Top user doc nhieu")
plt.savefig("user.png")
plt.close()

# ================= RESAMPLING =================

history_ts = history.set_index('ngay_bat_dau')

monthly_reads = history_ts['book_id'].resample('M').count()
daily_reads = history_ts['book_id'].resample('D').count().fillna(0)

# Monthly
plt.figure()
monthly_reads.plot(marker='o')
plt.title("Doc theo thang")
plt.savefig("monthly.png")
plt.close()

# Daily
plt.figure(figsize=(12,5))
daily_reads.plot()
plt.title("Doc theo ngay")
plt.savefig("daily.png")
plt.close()

# Smoothing
rolling = daily_reads.rolling(30).mean()

plt.figure(figsize=(12,5))
daily_reads.plot(alpha=0.5, label="Daily")
rolling.plot(label="Rolling 30")
plt.legend()
plt.title("Xu huong doc")
plt.savefig("trend.png")
plt.close()

# ================= TIME INDEX =================

history_period = history_ts.to_period('M')
period_reads = history_period.groupby(history_period.index).size()

plt.figure()
period_reads.plot(marker='o')
plt.title("Doc theo period")
plt.savefig("period.png")
plt.close()

# ================= REGRESSION =================

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

df_model = monthly_reads.reset_index()
df_model.columns = ['date','read_count']

df_model['time_index'] = np.arange(len(df_model))
df_model['month'] = df_model['date'].dt.month

train_size = int(len(df_model)*0.8)

train = df_model[:train_size]
test = df_model[train_size:]

X_train = train[['time_index','month']]
y_train = train['read_count']

X_test = test[['time_index','month']]
y_test = test['read_count']

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# Metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mape = np.mean(np.abs((y_test - y_pred)/(y_test+1e-6)))*100

print("\n===== METRICS =====")
print("MAE:", mae)
print("RMSE:", rmse)
print("MAPE:", mape)

# Forecast
plt.figure()
plt.plot(train['date'], train['read_count'], label='Train')
plt.plot(test['date'], y_test, label='Actual')
plt.plot(test['date'], y_pred, label='Predict')
plt.legend()
plt.title("Forecast")
plt.savefig("forecast.png")
plt.close()

# ================= TRENDING =================

df_model['MA'] = df_model['read_count'].rolling(3).mean()
df_model['EMA'] = df_model['read_count'].ewm(span=3).mean()

plt.figure()
plt.plot(df_model['date'], df_model['read_count'], label='Original')
plt.plot(df_model['date'], df_model['MA'], label='MA')
plt.plot(df_model['date'], df_model['EMA'], label='EMA')
plt.legend()
plt.title("Trending")
plt.savefig("trend_analysis.png")
plt.close()

# ================= PANDAS (WES) =================

genre_stats = books.groupby('the_loai')['diem_danh_gia'].agg(['mean','count'])
print("\nGenre stats:\n", genre_stats)

pivot = df.pivot_table(
    values='rating_book',
    index='the_loai',
    columns='user_id',
    aggfunc='mean'
)

print("\nPivot:\n", pivot.head())

print("🔥 DONE CLEAN CODE")