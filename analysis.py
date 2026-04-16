import pandas as pd
import matplotlib
matplotlib.use('Agg')  # tránh lỗi treo

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# ===== STYLE =====
sns.set_theme(style="whitegrid", palette="Set2")
plt.rcParams.update({
    "figure.figsize": (10,6),
    "axes.titlesize": 16,
    "axes.labelsize": 12
})

# ===== LOAD =====
users = pd.read_csv("users.csv")
books = pd.read_csv("books.csv")
history = pd.read_csv("cleaned_reading_history.csv")

print("✅ Load dữ liệu thành công")



# tạo số sách mua từ history (CHẮC CHẮN CÓ)
book_count = history.groupby('user_id')['book_id'].nunique().reset_index()
book_count.columns = ['user_id', 'so_luong_sach_da_mua']

# merge vào users (overwrite luôn)
users = users.drop(columns=[col for col in users.columns if col == 'so_luong_sach_da_mua'], errors='ignore')
users = users.merge(book_count, on='user_id', how='left').fillna(0)

# đảm bảo có ngày sinh
if 'ngay_sinh' not in users.columns:
    users['ngay_sinh'] = pd.to_datetime("2000-01-01")

# ===== MERGE SAU KHI FIX =====
df = history.merge(books, on="book_id").merge(users, on="user_id")

df = df.rename(columns={
    "diem_danh_gia_x": "rating_user",
    "diem_danh_gia_y": "rating_book"
})

# ================= GENRE =================
plt.figure()
sns.countplot(data=books, x="the_loai",
              order=books['the_loai'].value_counts().index)
plt.title("So luong sach theo the loai")
plt.xticks(rotation=30)
plt.savefig("genre.png")
plt.close()

# ================= TOP RATING =================
top_books = books.sort_values(by="diem_danh_gia", ascending=False).head(10)

plt.figure()
sns.barplot(data=top_books, x="diem_danh_gia", y="ten_sach")
plt.title("Top 10 sach rating cao")
plt.savefig("top_rating.png")
plt.close()

# ================= STATUS =================
status_count = history['trang_thai'].value_counts().reset_index()
status_count.columns = ["trang_thai", "count"]

plt.figure()
sns.barplot(data=status_count, x="trang_thai", y="count")
plt.title("Trang thai doc sach")
plt.savefig("status.png")
plt.close()

# ================= USER =================
user_read = history['user_id'].value_counts().head(10).reset_index()
user_read.columns = ["user_id", "count"]

plt.figure()
sns.barplot(data=user_read, x="user_id", y="count")
plt.title("Top user doc nhieu")
plt.savefig("user.png")
plt.close()

# ================= TIME =================
# ================= TIME (FIX CHUẨN) =================

# đảm bảo convert datetime trước khi dùng .dt
history['ngay_bat_dau'] = pd.to_datetime(history['ngay_bat_dau'], errors='coerce')

# loại bỏ giá trị lỗi
history = history.dropna(subset=['ngay_bat_dau'])

# tính theo tháng
time_read = history.groupby(history['ngay_bat_dau'].dt.month).size()

# đảm bảo đủ 12 tháng
months = pd.DataFrame({"month": range(1,13)})
time_df = pd.DataFrame({
    "month": time_read.index,
    "count": time_read.values
})

time_df = months.merge(time_df, on="month", how="left").fillna(0)

# vẽ
plt.figure()
sns.lineplot(data=time_df, x="month", y="count", marker="o")
sns.scatterplot(data=time_df, x="month", y="count")
plt.title("Xu huong doc theo thang")
plt.xticks(range(1,13))
plt.savefig("time.png")
plt.close()

# ================= RATING =================
df['rating_group'] = pd.cut(
    df['rating_book'],
    bins=[0, 3.5, 4.0, 4.5, 5],
    labels=["Thap", "Trung binh", "Cao", "Rat cao"]
)

plt.figure()
df['rating_group'].value_counts().plot.pie(autopct='%1.1f%%')
plt.title("Phan bo rating")
plt.ylabel("")
plt.savefig("rating_pie.png")
plt.close()

# ================= PIE EXTRA =================
plt.figure()
books['the_loai'].value_counts().plot.pie(autopct='%1.1f%%')
plt.title("Ty le the loai")
plt.savefig("pie_genre.png")
plt.close()

plt.figure()
history['trang_thai'].value_counts().plot.pie(autopct='%1.1f%%')
plt.title("Ty le trang thai")
plt.savefig("pie_status.png")
plt.close()

plt.figure()
history['user_id'].value_counts().head(5).plot.pie(autopct='%1.1f%%')
plt.title("Top user")
plt.savefig("pie_user.png")
plt.close()

# ================= AGE =================
df['so_luong_sach_da_mua'] = df.groupby('user_id')['book_id'].transform('nunique')
# ===== AGE =====
users['ngay_sinh'] = pd.to_datetime(users['ngay_sinh'], errors='coerce')
users['tuoi'] = 2025 - users['ngay_sinh'].dt.year

users['age_group'] = pd.cut(
    users['tuoi'],
    bins=[0, 18, 25, 35, 50],
    labels=["<18", "18-25", "25-35", "35+"]
)

# merge lại CHỈ để lấy age_group
df_age = df.merge(
    users[['user_id','tuoi','age_group']],
    on='user_id',
    how='left'
)

# đảm bảo không null
df_age = df_age.dropna(subset=['age_group'])

plt.figure()
sns.barplot(data=df_age, x="age_group", y="so_luong_sach_da_mua")
plt.title("So sach mua theo tuoi")
plt.savefig("age_books.png")
plt.close()

plt.figure()
sns.violinplot(data=df_age, x="age_group", y="rating_book")
plt.title("Rating theo tuoi")
plt.savefig("age_rating.png")
plt.close()

plt.figure()
sns.countplot(data=df_age, x="age_group")
plt.title("So luot doc theo tuoi")
plt.savefig("age_read.png")
plt.close()

plt.figure()
sns.regplot(data=df_age, x="tuoi", y="so_luong_sach_da_mua")
plt.title("Tuoi vs sach mua")
plt.savefig("age_trend.png")
plt.close()

# ================= HEATMAP =================
pivot = history.pivot_table(
    index="user_id",
    columns="book_id",
    values="diem_danh_gia"
).fillna(0)

plt.figure(figsize=(12,6))
sns.heatmap(pivot, cmap="YlOrRd")
plt.title("Heatmap User Book")
plt.savefig("heatmap.png")
plt.close()

print("🔥 HOÀN THÀNH FULL KHÔNG LỖI!")