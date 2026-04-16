import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ===== STYLE HIỆN ĐẠI =====
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

# ===== MERGE =====
df = history.merge(books, on="book_id").merge(users, on="user_id")

df = df.rename(columns={
    "diem_danh_gia_x": "rating_user",
    "diem_danh_gia_y": "rating_book"
})

# ================= 1. GENRE (ĐẸP HƠN) =================
plt.figure()
sns.countplot(
    data=df,
    x="the_loai",
    order=df['the_loai'].value_counts().index,
    edgecolor="black"
)
plt.title("📚 Số lượt đọc theo thể loại")
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig("modern_genre.png")

# ================= 2. BOXPLOT =================
plt.figure()
sns.boxplot(
    data=df,
    x="the_loai",
    y="rating_book",
    palette="pastel"
)
sns.stripplot(
    data=df,
    x="the_loai",
    y="rating_book",
    color="black",
    size=4,
    alpha=0.5
)
plt.title("⭐ Phân bố rating theo thể loại")
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig("modern_boxplot.png")

# ================= 3. SCATTER PRO =================
plt.figure()
sns.scatterplot(
    data=df,
    x="gia_sach",
    y="rating_book",
    hue="the_loai",
    size="rating_book",
    sizes=(50,200),
    alpha=0.8
)
plt.title("💰 Giá sách vs Rating")
plt.tight_layout()
plt.savefig("modern_scatter.png")

# ================= 4. HIST + KDE =================
plt.figure()
sns.histplot(
    df['rating_book'],
    bins=10,
    kde=True,
    color="skyblue"
)
plt.title("📊 Phân phối rating")
plt.tight_layout()
plt.savefig("modern_hist.png")

# ================= 5. HEATMAP PRO =================
pivot = history.pivot_table(
    index="user_id",
    columns="book_id",
    values="diem_danh_gia"
).fillna(0)

plt.figure(figsize=(12,6))
sns.heatmap(
    pivot,
    cmap="YlOrRd",
    linewidths=0.3,
    linecolor="gray",
    cbar_kws={'label': 'Rating'}
)
plt.title("🔥 Heatmap User - Book")
plt.tight_layout()
plt.savefig("modern_heatmap.png")

# ================= 6. AGE VS BOOKS =================
users['ngay_sinh'] = pd.to_datetime(users['ngay_sinh'])
users['tuoi'] = 2025 - users['ngay_sinh'].dt.year

plt.figure()
sns.regplot(
    data=users,
    x="tuoi",
    y="so_luong_sach_da_mua",
    scatter_kws={"s":80},
    line_kws={"color":"red"}
)
plt.title("👤 Tuổi vs số sách mua")
plt.tight_layout()
plt.savefig("modern_age.png")

print("🔥 Biểu đồ modern đã tạo xong!")