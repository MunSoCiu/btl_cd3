"""
generate_charts.py
Tạo toàn bộ biểu đồ đúng từ 14 dataset thực tế.
Chạy: python3 generate_charts.py
Output: thư mục charts/
"""

import os, warnings
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

warnings.filterwarnings('ignore')
os.makedirs('charts', exist_ok=True)

# ── Palette & style ─────────────────────────────────────────────────────────
TEAL   = '#4ABDAC'
CORAL  = '#FC4A1A'
GOLD   = '#F7B733'
NAVY   = '#1B2A4A'
SLATE  = '#6C7A89'
MINT   = '#A8E6CF'
PEACH  = '#FFAAA5'
LAVENDER = '#B0A8D8'
PIE_COLORS = [TEAL, CORAL, GOLD, NAVY, SLATE, MINT, PEACH, LAVENDER,
              '#E8D5B7','#C9E4DE','#F5CBA7','#AED6F1']

sns.set_theme(style='whitegrid', font='DejaVu Sans')
plt.rcParams.update({
    'figure.dpi': 130,
    'axes.titlesize': 14,
    'axes.titleweight': 'bold',
    'axes.labelsize': 11,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

def fmt_vnd(x, _=None):
    if x >= 1e9:  return f'{x/1e9:.1f}B'
    if x >= 1e6:  return f'{x/1e6:.0f}M'
    return f'{x/1e3:.0f}K'

def savefig(name):
    path = f'charts/{name}'
    plt.tight_layout()
    plt.savefig(path, bbox_inches='tight')
    plt.close('all')
    print(f'  ✓ {name}')

# ── Load data ────────────────────────────────────────────────────────────────
BASE = os.path.dirname(os.path.abspath(__file__))

users = pd.read_csv(os.path.join(BASE, '../data/users.csv'))
books = pd.read_csv(os.path.join(BASE, '../data/books.csv'))
history = pd.read_csv(os.path.join(BASE, '../data/cleaned_reading_history.csv'))
prefs = pd.read_csv(os.path.join(BASE, '../data/user_preferences.csv'))
sessions = pd.read_csv(os.path.join(BASE, '../data/reading_sessions.csv'))
fact     = pd.read_csv(os.path.join(BASE, '../data/fact_giao_dich.csv'))
rev_city = pd.read_csv(os.path.join(BASE, '../data/revenue_by_city.csv'))
rev_genre= pd.read_csv(os.path.join(BASE, '../data/revenue_by_genre.csv'))
rev_month= pd.read_csv(os.path.join(BASE, '../data/revenue_by_month.csv'))
rev_qtr  = pd.read_csv(os.path.join(BASE, '../data/revenue_by_quarter.csv'))
rev_user = pd.read_csv(os.path.join(BASE, '../data/revenue_by_user.csv'))
rev_year = pd.read_csv(os.path.join(BASE, '../data/revenue_by_year.csv'))
cohort   = pd.read_csv(os.path.join(BASE, '../data/cohort_revenue.csv'))
kpi      = pd.read_csv(os.path.join(BASE, '../data/kpi_summary.csv'))

# ── Prepare shared columns ───────────────────────────────────────────────────
users['ngay_sinh'] = pd.to_datetime(users['ngay_sinh'], format='mixed')
users['tuoi'] = 2025 - users['ngay_sinh'].dt.year
def age_grp(a):
    if a < 18: return '<18'
    elif a <= 24: return '18-24'
    elif a <= 35: return '25-35'
    elif a <= 45: return '36-45'
    else: return '46+'
users['age_group'] = users['tuoi'].apply(age_grp)

history['ngay_bat_dau'] = pd.to_datetime(history['ngay_bat_dau'], format='mixed')
history['thang'] = history['ngay_bat_dau'].dt.month
history['nam']   = history['ngay_bat_dau'].dt.year
history['thang_nam'] = history['ngay_bat_dau'].dt.to_period('M').astype(str)

fact['ngay_giao_dich'] = pd.to_datetime(fact['ngay_giao_dich'])
success = fact[fact['trang_thai_giao_dich'] == 'Thành công'].copy()

print("═"*55)
print("  GENERATING CHARTS")
print("═"*55)

# ════════════════════════════════════════════════════════════
# NHÓM 1 – NGƯỜI DÙNG & HÀNH VI ĐỌC
# ════════════════════════════════════════════════════════════
print("\n[GROUP 1] Người dùng & hành vi đọc")

# 01 – Số sách trung bình theo nhóm tuổi (BAR + ERRORBAR)
age_order = ['18-24','25-35','36-45']
agg = users.groupby('age_group')['so_luong_sach_da_mua'].agg(['mean','std']).reindex(age_order)
fig, ax = plt.subplots(figsize=(8,5))
bars = ax.bar(agg.index, agg['mean'], color=[TEAL,CORAL,GOLD], width=0.5, zorder=3)
ax.errorbar(agg.index, agg['mean'], yerr=agg['std'],
            fmt='none', color=NAVY, capsize=6, linewidth=2, zorder=4)
for b,v in zip(bars, agg['mean']):
    ax.text(b.get_x()+b.get_width()/2, v+0.5, f'{v:.1f}', ha='center', fontsize=11, fontweight='bold')
ax.set_title('Số Sách Trung Bình Theo Nhóm Tuổi', pad=12)
ax.set_xlabel('Nhóm tuổi'); ax.set_ylabel('Số sách TB')
ax.set_ylim(0, agg['mean'].max()*1.35)
savefig('01_age_vs_books.png')

# 02 – Số lượt đọc theo nhóm tuổi (từ reading_sessions + users)
sess_u = sessions.merge(users[['user_id','age_group']], left_on='user_id', right_on='user_id')
age_sess = sess_u.groupby('age_group').size().reindex(age_order)
fig, ax = plt.subplots(figsize=(8,5))
bars = ax.bar(age_sess.index, age_sess.values, color=[TEAL,CORAL,GOLD], width=0.5, zorder=3)
for b,v in zip(bars, age_sess.values):
    ax.text(b.get_x()+b.get_width()/2, v+0.5, str(v), ha='center', fontsize=11, fontweight='bold')
ax.set_title('Số Lượt Đọc Theo Nhóm Tuổi', pad=12)
ax.set_xlabel('Nhóm tuổi'); ax.set_ylabel('Số lượt đọc')
savefig('02_age_vs_reads.png')

# 03 – Tuổi vs Số sách mua (scatter toàn bộ 500 users)
fig, ax = plt.subplots(figsize=(9,5))
sc = ax.scatter(users['tuoi'], users['so_luong_sach_da_mua'],
                alpha=0.45, color=TEAL, edgecolors='white', linewidths=0.4, s=40, zorder=3)
m, b = np.polyfit(users['tuoi'], users['so_luong_sach_da_mua'], 1)
xs = np.linspace(users['tuoi'].min(), users['tuoi'].max(), 100)
ax.plot(xs, m*xs+b, color=CORAL, linewidth=2.2, zorder=4, label=f'Trend (slope={m:.2f})')
ax.set_title('Tuổi vs Số Sách Mua (500 users)', pad=12)
ax.set_xlabel('Tuổi'); ax.set_ylabel('Số sách đã mua')
ax.legend(fontsize=10)
savefig('03_age_trend_scatter.png')

# 04 – Trạng thái đọc sách (BAR)
st = history['trang_thai'].value_counts().sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(8,5))
colors_st = [TEAL, CORAL, GOLD, SLATE]
bars = ax.bar(st.index, st.values, color=colors_st[:len(st)], width=0.55, zorder=3)
for b,v in zip(bars, st.values):
    ax.text(b.get_x()+b.get_width()/2, v+0.5, str(v), ha='center', fontsize=11, fontweight='bold')
ax.set_title('Trạng Thái Đọc Sách', pad=12)
ax.set_xlabel('Trạng thái'); ax.set_ylabel('Số lượng')
savefig('04_reading_status_bar.png')

# 05 – Tỷ lệ trạng thái đọc (PIE)
fig, ax = plt.subplots(figsize=(7,6))
wedges, texts, autotexts = ax.pie(
    st.values, labels=st.index, autopct='%1.1f%%',
    colors=PIE_COLORS[:len(st)], startangle=140,
    wedgeprops={'edgecolor':'white','linewidth':2})
for t in autotexts: t.set_fontsize(11); t.set_fontweight('bold')
ax.set_title('Tỷ Lệ Trạng Thái Đọc', pad=12)
savefig('05_pie_reading_status.png')

# ===== DATA =====
top_read = history['user_id'].value_counts().head(10)
top_read_df = top_read.reset_index()
top_read_df.columns = ['user_id','count']

top_read_df = top_read_df.merge(
    users[['user_id','ho_ten']],
    left_on='user_id',
    right_on='user_id',
    how='left'
)

top_read_df['ho_ten'] = top_read_df['ho_ten'].fillna('Unknown')

# Label đúng
top_read_df['label'] = top_read_df['ho_ten'] + ' (#' + top_read_df['user_id'].astype(str) + ')'

# ===== PLOT =====
fig, ax = plt.subplots(figsize=(9,5))

bars = ax.barh(
    top_read_df['label'],
    top_read_df['count'],
    color=TEAL,
    zorder=3
)

# Hiển thị số
for b in bars:
    ax.text(
        b.get_width()+0.2,
        b.get_y()+b.get_height()/2,
        str(int(b.get_width())),
        va='center',
        fontsize=10,
        fontweight='bold'
    )

ax.set_title('Top 10 Users Đọc Nhiều Nhất', pad=12)
ax.set_xlabel('Số lần đọc')
ax.invert_yaxis()

savefig('06_top_users_reads.png')

# 07 – Xu hướng đọc theo tháng (LINE – toàn bộ history)
monthly_reads = history.groupby('thang').size().reset_index(name='count')
fig, ax = plt.subplots(figsize=(10,5))
ax.plot(monthly_reads['thang'], monthly_reads['count'],
        color=TEAL, linewidth=2.5, marker='o', markersize=7, zorder=3)
ax.fill_between(monthly_reads['thang'], monthly_reads['count'], alpha=0.15, color=TEAL)
ax.set_xticks(range(1,13))
ax.set_xticklabels(['T1','T2','T3','T4','T5','T6','T7','T8','T9','T10','T11','T12'])
ax.set_title('Xu Hướng Đọc Theo Tháng (tổng hợp)', pad=12)
ax.set_xlabel('Tháng'); ax.set_ylabel('Số lượt đọc')
for x,y in zip(monthly_reads['thang'], monthly_reads['count']):
    ax.annotate(str(y), (x,y), textcoords='offset points', xytext=(0,8),
                ha='center', fontsize=9, fontweight='bold')
savefig('07_monthly_reads_trend.png')

# 08 – Heatmap User-Book Rating (user 1-20, book 1-20)
pivot = history[(history['user_id'].between(1,20)) & (history['book_id'].between(1,20))]\
        .pivot_table(index='user_id', columns='book_id', values='diem_danh_gia', fill_value=np.nan)
fig, ax = plt.subplots(figsize=(12,7))
sns.heatmap(pivot, cmap='YlOrRd', ax=ax, linewidths=0.3,
            cbar_kws={'label':'Rating'}, vmin=1, vmax=5)
ax.set_title('Heatmap Rating – User (1–20) × Book (1–20)', pad=12)
ax.set_xlabel('book_id'); ax.set_ylabel('user_id')
savefig('08_heatmap_user_book.png')

# 09 – Phân bố rating người dùng (từ history)
fig, ax = plt.subplots(figsize=(8,5))
ax.hist(history['diem_danh_gia'].dropna(), bins=10, color=TEAL,
        edgecolor='white', linewidth=0.8, zorder=3)
ax.set_title('Phân Phối Điểm Đánh Giá (Lịch Sử Đọc)', pad=12)
ax.set_xlabel('Điểm đánh giá'); ax.set_ylabel('Số lượng')
savefig('09_rating_distribution_history.png')

# 10 – Pie Top users hoạt động nhiều (top 5)
top5 = history['user_id'].value_counts().head(5)
labels5 = [f'User {uid}' for uid in top5.index]
fig, ax = plt.subplots(figsize=(7,6))
wedges,texts,autotexts = ax.pie(top5.values, labels=labels5, autopct='%1.1f%%',
    colors=PIE_COLORS[:5], startangle=140,
    wedgeprops={'edgecolor':'white','linewidth':2})
for t in autotexts: t.set_fontsize(11); t.set_fontweight('bold')
ax.set_title('Top 5 Users Hoạt Động Nhiều Nhất', pad=12)
savefig('10_pie_top_users.png')

# ════════════════════════════════════════════════════════════
# NHÓM 2 – SÁCH
# ════════════════════════════════════════════════════════════
print("\n[GROUP 2] Sách")

# 11 – Số lượng sách theo thể loại
genre_cnt = books['the_loai'].value_counts().sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(10,5))
bars = ax.bar(genre_cnt.index, genre_cnt.values, color=PIE_COLORS[:len(genre_cnt)], width=0.6, zorder=3)
for b,v in zip(bars, genre_cnt.values):
    ax.text(b.get_x()+b.get_width()/2, v+0.1, str(v), ha='center', fontsize=10, fontweight='bold')
ax.set_title('Số Lượng Sách Theo Thể Loại', pad=12)
ax.set_xlabel('Thể loại'); ax.set_ylabel('Số sách')
plt.xticks(rotation=20, ha='right')
savefig('11_books_by_genre.png')

# 12 – Pie tỷ lệ thể loại
fig, ax = plt.subplots(figsize=(8,7))
wedges,texts,autotexts = ax.pie(
    genre_cnt.values, labels=genre_cnt.index, autopct='%1.1f%%',
    colors=PIE_COLORS[:len(genre_cnt)], startangle=140,
    wedgeprops={'edgecolor':'white','linewidth':2})
for t in autotexts: t.set_fontsize(9); t.set_fontweight('bold')
ax.set_title('Tỷ Lệ Sách Theo Thể Loại', pad=12)
savefig('12_pie_genre.png')

# 13 – Top 10 sách rating cao nhất
top10 = books.nlargest(10, 'diem_danh_gia')[['ten_sach','diem_danh_gia','the_loai']]
fig, ax = plt.subplots(figsize=(10,6))
colors13 = [TEAL if i%2==0 else CORAL for i in range(len(top10))]
bars = ax.barh(top10['ten_sach'], top10['diem_danh_gia'], color=colors13, zorder=3)
for b in bars:
    ax.text(b.get_width()+0.01, b.get_y()+b.get_height()/2,
            f'{b.get_width():.1f}', va='center', fontsize=10, fontweight='bold')
ax.set_xlim(0, top10['diem_danh_gia'].max()*1.12)
ax.set_title('Top 10 Sách Có Điểm Đánh Giá Cao Nhất', pad=12)
ax.set_xlabel('Điểm đánh giá')
ax.invert_yaxis()
savefig('13_top10_books_rating.png')

# 14 – Phân phối rating sách (books.csv)
fig, ax = plt.subplots(figsize=(8,5))
ax.hist(books['diem_danh_gia'], bins=15, color=TEAL, edgecolor='white', linewidth=0.8, zorder=3, density=False)
ax.set_title('Phân Phối Điểm Đánh Giá Sách (books.csv)', pad=12)
ax.set_xlabel('Điểm đánh giá (diem_danh_gia)'); ax.set_ylabel('Số sách')
ax.set_xlim(3.0, 5.2)
savefig('14_book_rating_hist.png')

# 15 – Pie phân bố mức rating
def rg(r):
    if r < 3.5: return 'Thấp (<3.5)'
    elif r < 4.0: return 'TB (3.5–4.0)'
    elif r < 4.5: return 'Cao (4.0–4.5)'
    else: return 'Rất cao (≥4.5)'
books['rating_group'] = books['diem_danh_gia'].apply(rg)
rg_cnt = books['rating_group'].value_counts()
fig, ax = plt.subplots(figsize=(7,6))
wedges,texts,autotexts = ax.pie(rg_cnt.values, labels=rg_cnt.index, autopct='%1.1f%%',
    colors=[CORAL,GOLD,TEAL,NAVY], startangle=140,
    wedgeprops={'edgecolor':'white','linewidth':2})
for t in autotexts: t.set_fontsize(11); t.set_fontweight('bold')
ax.set_title('Phân Bố Mức Rating Sách', pad=12)
savefig('15_pie_rating_group.png')

# 16 – Boxplot rating theo thể loại
fig, ax = plt.subplots(figsize=(11,6))
order_box = books.groupby('the_loai')['diem_danh_gia'].median().sort_values(ascending=False).index
sns.boxplot(data=books, x='the_loai', y='diem_danh_gia',
            order=order_box, palette='Set2', ax=ax, zorder=3)
ax.set_title('Phân Bố Rating Theo Thể Loại (books.csv)', pad=12)
ax.set_xlabel('Thể loại'); ax.set_ylabel('Điểm đánh giá')
plt.xticks(rotation=20, ha='right')
savefig('16_boxplot_rating_by_genre.png')

# 17 – Scatter Giá sách vs Rating (toàn bộ 100 sách)
fig, ax = plt.subplots(figsize=(9,6))
genre_list = books['the_loai'].unique().tolist()
cmap = dict(zip(genre_list, PIE_COLORS[:len(genre_list)]))
for genre, grp in books.groupby('the_loai'):
    ax.scatter(grp['gia_sach'], grp['diem_danh_gia'],
               color=cmap[genre], s=grp['diem_danh_gia']*18,
               alpha=0.75, edgecolors='white', linewidths=0.5,
               label=genre, zorder=3)
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f'{int(x/1000)}K'))
ax.set_title('Giá Sách vs Điểm Đánh Giá (100 sách)', pad=12)
ax.set_xlabel('Giá sách (VNĐ)'); ax.set_ylabel('Điểm đánh giá')
ax.legend(title='Thể loại', bbox_to_anchor=(1.01,1), fontsize=9)
savefig('17_scatter_price_rating.png')

# 18 – Số lượt đọc theo thể loại
hist_books = history.merge(books[['book_id','the_loai']], on='book_id')
genre_reads = hist_books['the_loai'].value_counts().sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(10,5))
bars = ax.bar(genre_reads.index, genre_reads.values,
              color=PIE_COLORS[:len(genre_reads)], width=0.6, zorder=3)
for b,v in zip(bars, genre_reads.values):
    ax.text(b.get_x()+b.get_width()/2, v+1, str(v), ha='center', fontsize=10, fontweight='bold')
ax.set_title('Số Lượt Đọc Theo Thể Loại', pad=12)
ax.set_xlabel('Thể loại'); ax.set_ylabel('Số lượt')
plt.xticks(rotation=20, ha='right')
savefig('18_reads_by_genre.png')

# 19 – Phân phối giá sách
fig, ax = plt.subplots(figsize=(8,5))
ax.hist(books['gia_sach'], bins=15, color=CORAL, edgecolor='white', linewidth=0.8, zorder=3)
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f'{int(x/1000)}K'))
ax.set_title('Phân Phối Giá Sách (VNĐ)', pad=12)
ax.set_xlabel('Giá sách'); ax.set_ylabel('Số sách')
savefig('19_book_price_dist.png')

# ════════════════════════════════════════════════════════════
# NHÓM 3 – DOANH THU & GIAO DỊCH
# ════════════════════════════════════════════════════════════
print("\n[GROUP 3] Doanh thu & giao dịch")

# 20 – Doanh thu theo năm (BAR)
fig, ax = plt.subplots(figsize=(8,5))
bars = ax.bar(rev_year['nam'].astype(str), rev_year['tong_doanh_thu'],
              color=[TEAL,CORAL,GOLD,NAVY], width=0.55, zorder=3)
for b,v in zip(bars, rev_year['tong_doanh_thu']):
    ax.text(b.get_x()+b.get_width()/2, v+3e7, fmt_vnd(v),
            ha='center', fontsize=11, fontweight='bold')
ax.yaxis.set_major_formatter(mticker.FuncFormatter(fmt_vnd))
ax.set_title('Tổng Doanh Thu Theo Năm', pad=12)
ax.set_xlabel('Năm'); ax.set_ylabel('Doanh thu (VNĐ)')
savefig('20_revenue_by_year.png')

# 21 – Tăng trưởng YoY
fig, ax = plt.subplots(figsize=(8,5))
yoy = rev_year[rev_year['tang_truong_yoy']!=0]
colors21 = [TEAL if v>0 else CORAL for v in yoy['tang_truong_yoy']]
bars = ax.bar(yoy['nam'].astype(str), yoy['tang_truong_yoy'], color=colors21, width=0.5, zorder=3)
for b,v in zip(bars, yoy['tang_truong_yoy']):
    ax.text(b.get_x()+b.get_width()/2, v+0.3, f'{v:.1f}%', ha='center', fontsize=11, fontweight='bold')
ax.axhline(0, color=SLATE, linewidth=1)
ax.set_title('Tăng Trưởng Doanh Thu YoY (%)', pad=12)
ax.set_xlabel('Năm'); ax.set_ylabel('Tăng trưởng (%)')
savefig('21_revenue_yoy_growth.png')

# 22 – Doanh thu theo tháng (LINE)
fig, ax = plt.subplots(figsize=(14,5))
ax.plot(rev_month['thang_nam'], rev_month['tong_doanh_thu'],
        color=TEAL, linewidth=2, marker='o', markersize=4, zorder=3)
ax.fill_between(range(len(rev_month)), rev_month['tong_doanh_thu'], alpha=0.12, color=TEAL)
ax.set_xticks(range(0, len(rev_month), 6))
ax.set_xticklabels(rev_month['thang_nam'].iloc[::6], rotation=30, ha='right')
ax.yaxis.set_major_formatter(mticker.FuncFormatter(fmt_vnd))
ax.set_title('Doanh Thu Theo Tháng (2022–2025)', pad=12)
ax.set_xlabel('Tháng'); ax.set_ylabel('Doanh thu (VNĐ)')
savefig('22_revenue_by_month.png')

# 23 – Doanh thu lũy kế
fig, ax = plt.subplots(figsize=(14,5))
ax.plot(rev_month['thang_nam'], rev_month['doanh_thu_luy_ke'],
        color=CORAL, linewidth=2.5, marker='', zorder=3)
ax.fill_between(range(len(rev_month)), rev_month['doanh_thu_luy_ke'], alpha=0.12, color=CORAL)
ax.set_xticks(range(0, len(rev_month), 6))
ax.set_xticklabels(rev_month['thang_nam'].iloc[::6], rotation=30, ha='right')
ax.yaxis.set_major_formatter(mticker.FuncFormatter(fmt_vnd))
ax.set_title('Doanh Thu Lũy Kế (2022–2025)', pad=12)
ax.set_xlabel('Tháng'); ax.set_ylabel('Doanh thu lũy kế (VNĐ)')
savefig('23_revenue_cumulative.png')

# 24 – Doanh thu theo quý (grouped bar)
fig, ax = plt.subplots(figsize=(14,5))
qtr_colors = {'Q1':TEAL,'Q2':CORAL,'Q3':GOLD,'Q4':NAVY}
for quy, grp in rev_qtr.groupby('quy'):
    ax.bar(grp['nam'].astype(str), grp['tong_doanh_thu'],
           color=qtr_colors[quy], label=quy, width=0.2,
           align='center', zorder=3)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(fmt_vnd))
ax.set_title('Doanh Thu Theo Quý (2022–2025)', pad=12)
ax.set_xlabel('Năm'); ax.set_ylabel('Doanh thu (VNĐ)')
ax.legend(title='Quý')
savefig('24_revenue_by_quarter.png')

# 25 – Doanh thu theo thành phố
fig, ax = plt.subplots(figsize=(9,5))
rc = rev_city.sort_values('tong_doanh_thu', ascending=True)
colors25 = [TEAL,CORAL,GOLD,NAVY,SLATE]
bars = ax.barh(rc['thanh_pho'], rc['tong_doanh_thu'], color=colors25, zorder=3)
for b in bars:
    ax.text(b.get_width()+5e7, b.get_y()+b.get_height()/2,
            fmt_vnd(b.get_width()), va='center', fontsize=10, fontweight='bold')
ax.xaxis.set_major_formatter(mticker.FuncFormatter(fmt_vnd))
ax.set_title('Doanh Thu Theo Thành Phố', pad=12)
ax.set_xlabel('Doanh thu (VNĐ)')
savefig('25_revenue_by_city.png')

# 26 – Pie doanh thu theo thành phố
fig, ax = plt.subplots(figsize=(7,6))
wedges,texts,autotexts = ax.pie(
    rev_city['tong_doanh_thu'], labels=rev_city['thanh_pho'], autopct='%1.1f%%',
    colors=PIE_COLORS[:5], startangle=140,
    wedgeprops={'edgecolor':'white','linewidth':2})
for t in autotexts: t.set_fontsize(11); t.set_fontweight('bold')
ax.set_title('Tỷ Lệ Doanh Thu Theo Thành Phố', pad=12)
savefig('26_pie_revenue_by_city.png')

# 27 – Doanh thu theo thể loại
fig, ax = plt.subplots(figsize=(10,5))
rg2 = rev_genre.sort_values('tong_doanh_thu', ascending=False)
bars = ax.bar(rg2['the_loai'], rg2['tong_doanh_thu'],
              color=PIE_COLORS[:len(rg2)], width=0.6, zorder=3)
for b,v in zip(bars, rg2['tong_doanh_thu']):
    ax.text(b.get_x()+b.get_width()/2, v+1e7, fmt_vnd(v),
            ha='center', fontsize=9, fontweight='bold')
ax.yaxis.set_major_formatter(mticker.FuncFormatter(fmt_vnd))
ax.set_title('Doanh Thu Theo Thể Loại Sách', pad=12)
ax.set_xlabel('Thể loại'); ax.set_ylabel('Doanh thu (VNĐ)')
plt.xticks(rotation=20, ha='right')
savefig('27_revenue_by_genre.png')

# 28 – Phương thức thanh toán
pm = success['phuong_thuc_thanh_toan'].value_counts()
fig, ax = plt.subplots(figsize=(9,5))
bars = ax.barh(pm.index, pm.values, color=PIE_COLORS[:len(pm)], zorder=3)
for b in bars:
    ax.text(b.get_width()+50, b.get_y()+b.get_height()/2,
            f'{int(b.get_width()):,}', va='center', fontsize=10, fontweight='bold')
ax.set_title('Số Giao Dịch Theo Phương Thức Thanh Toán', pad=12)
ax.set_xlabel('Số giao dịch')
ax.invert_yaxis()
savefig('28_payment_method.png')

# 29 – Kênh mua hàng
ch = success['kenh_mua'].value_counts()
fig, ax = plt.subplots(figsize=(8,5))
wedges,texts,autotexts = ax.pie(
    ch.values, labels=ch.index, autopct='%1.1f%%',
    colors=PIE_COLORS[:4], startangle=140,
    wedgeprops={'edgecolor':'white','linewidth':2})
for t in autotexts: t.set_fontsize(11); t.set_fontweight('bold')
ax.set_title('Tỷ Lệ Kênh Mua Hàng', pad=12)
savefig('29_pie_channel.png')

# 30 – Tỷ lệ trạng thái giao dịch
ts = fact['trang_thai_giao_dich'].value_counts()
fig, ax = plt.subplots(figsize=(7,6))
explode = [0.04]*len(ts)
wedges,texts,autotexts = ax.pie(
    ts.values, labels=ts.index, autopct='%1.1f%%',
    colors=[TEAL,CORAL,GOLD], explode=explode, startangle=140,
    wedgeprops={'edgecolor':'white','linewidth':2})
for t in autotexts: t.set_fontsize(11); t.set_fontweight('bold')
ax.set_title('Tỷ Lệ Trạng Thái Giao Dịch', pad=12)
savefig('30_pie_transaction_status.png')

# 31 – Top 15 sách bán chạy nhất
top_books = success['ten_sach'].value_counts().head(15)
fig, ax = plt.subplots(figsize=(10,6))
colors31 = [TEAL if i < 5 else CORAL if i < 10 else GOLD for i in range(len(top_books))]
bars = ax.barh(top_books.index, top_books.values, color=colors31, zorder=3)
for b in bars:
    ax.text(b.get_width()+2, b.get_y()+b.get_height()/2,
            str(int(b.get_width())), va='center', fontsize=9, fontweight='bold')
ax.set_title('Top 15 Sách Bán Chạy Nhất', pad=12)
ax.set_xlabel('Số lượng giao dịch')
ax.invert_yaxis()
savefig('31_top15_best_sellers.png')

# ===== DATA =====
top15_u = rev_user.head(15).copy()

top15_u['ho_ten'] = top15_u['ho_ten'].fillna('Unknown')

top15_u['label'] = top15_u['ho_ten'] + ' (#' + top15_u['user_id'].astype(str) + ')'

# ===== PLOT =====
fig, ax = plt.subplots(figsize=(10,6))

bars = ax.barh(
    top15_u['label'],
    top15_u['tong_doanh_thu'],
    color=TEAL,
    zorder=3
)

for b in bars:
    ax.text(
        b.get_width()+2e5,
        b.get_y()+b.get_height()/2,
        fmt_vnd(b.get_width()),
        va='center',
        fontsize=9,
        fontweight='bold'
    )

ax.xaxis.set_major_formatter(mticker.FuncFormatter(fmt_vnd))
ax.set_title('Top 15 Users Doanh Thu Cao Nhất', pad=12)
ax.set_xlabel('Tổng doanh thu (VNĐ)')
ax.invert_yaxis()

savefig('32_top15_users_revenue.png')

# 33 – Số giao dịch theo tháng (BAR)
fig, ax = plt.subplots(figsize=(14,5))
bars = ax.bar(rev_month['thang_nam'], rev_month['so_giao_dich'],
              color=TEAL, width=0.7, zorder=3)
ax.set_xticks(range(0, len(rev_month), 6))
ax.set_xticklabels(rev_month['thang_nam'].iloc[::6], rotation=30, ha='right')
ax.set_title('Số Giao Dịch Theo Tháng', pad=12)
ax.set_xlabel('Tháng'); ax.set_ylabel('Số giao dịch')
savefig('33_transactions_by_month.png')

# 34 – Cohort Revenue heatmap (top 8 cohorts)
cohort['cohort_thang'] = cohort['cohort_thang'].astype(str)
cohort['tx_period']    = cohort['tx_period'].astype(str)
top_cohorts = cohort.groupby('cohort_thang')['tong_doanh_thu'].sum()\
                     .nlargest(8).index.tolist()
ch_pivot = cohort[cohort['cohort_thang'].isin(top_cohorts)]\
           .pivot_table(index='cohort_thang', columns='tx_period',
                        values='tong_doanh_thu', fill_value=0)
# Keep first 12 periods
ch_pivot = ch_pivot.iloc[:, :12]
fig, ax = plt.subplots(figsize=(14,6))
sns.heatmap(ch_pivot/1e6, cmap='YlOrRd', ax=ax, linewidths=0.3,
            fmt='.0f', annot=True, annot_kws={'size':8},
            cbar_kws={'label':'Doanh thu (triệu VNĐ)'})
ax.set_title('Cohort Revenue Heatmap (Top 8 cohorts, triệu VNĐ)', pad=12)
ax.set_xlabel('Tháng giao dịch'); ax.set_ylabel('Cohort tháng')
plt.xticks(rotation=30, ha='right')
savefig('34_cohort_heatmap.png')

# 35 – Tăng trưởng MoM
fig, ax = plt.subplots(figsize=(14,5))
mom = rev_month[rev_month['tang_truong_mom'] != 0]
colors35 = [TEAL if v > 0 else CORAL for v in mom['tang_truong_mom']]
ax.bar(mom['thang_nam'], mom['tang_truong_mom'], color=colors35, width=0.7, zorder=3)
ax.axhline(0, color=SLATE, linewidth=1)
ax.set_xticks(range(0, len(mom), 6))
ax.set_xticklabels(mom['thang_nam'].iloc[::6], rotation=30, ha='right')
ax.set_title('Tăng Trưởng Doanh Thu MoM (%)', pad=12)
ax.set_xlabel('Tháng'); ax.set_ylabel('Tăng trưởng (%)')
savefig('35_revenue_mom_growth.png')

# ════════════════════════════════════════════════════════════
# NHÓM 4 – SỞ THÍCH & SESSIONS
# ════════════════════════════════════════════════════════════
print("\n[GROUP 4] Sở thích & sessions")

# 36 – Thiết bị đọc
dev = prefs['thiet_bi_doc'].value_counts()
fig, ax = plt.subplots(figsize=(7,6))
wedges,texts,autotexts = ax.pie(
    dev.values, labels=dev.index, autopct='%1.1f%%',
    colors=PIE_COLORS[:len(dev)], startangle=140,
    wedgeprops={'edgecolor':'white','linewidth':2})
for t in autotexts: t.set_fontsize(11); t.set_fontweight('bold')
ax.set_title('Phân Bố Thiết Bị Đọc Sách', pad=12)
savefig('36_pie_device.png')

# 37 – Thời gian đọc ưa thích
time_pref = prefs['thoi_gian_doc'].value_counts()
fig, ax = plt.subplots(figsize=(8,5))
bars = ax.bar(time_pref.index, time_pref.values,
              color=[TEAL,CORAL,GOLD][:len(time_pref)], width=0.5, zorder=3)
for b,v in zip(bars, time_pref.values):
    ax.text(b.get_x()+b.get_width()/2, v+1, str(v), ha='center', fontsize=11, fontweight='bold')
ax.set_title('Thời Gian Đọc Ưa Thích', pad=12)
ax.set_xlabel('Thời gian'); ax.set_ylabel('Số người dùng')
savefig('37_reading_time_pref.png')

# 38 – Ngôn ngữ ưa thích
lang = prefs['ngon_ngu'].value_counts()
fig, ax = plt.subplots(figsize=(6,5))
bars = ax.bar(lang.index, lang.values,
              color=[TEAL,CORAL][:len(lang)], width=0.4, zorder=3)
for b,v in zip(bars, lang.values):
    ax.text(b.get_x()+b.get_width()/2, v+1, str(v), ha='center', fontsize=12, fontweight='bold')
ax.set_title('Ngôn Ngữ Đọc Ưa Thích', pad=12)
ax.set_xlabel('Ngôn ngữ'); ax.set_ylabel('Số người dùng')
savefig('38_language_pref.png')

# 39 – Phân phối thời gian đọc/session (phút)
fig, ax = plt.subplots(figsize=(8,5))
ax.hist(sessions['thoi_gian_doc_phut'], bins=12, color=MINT,
        edgecolor='white', linewidth=0.8, zorder=3)
ax.set_title('Phân Phối Thời Gian Đọc Mỗi Session (phút)', pad=12)
ax.set_xlabel('Thời gian (phút)'); ax.set_ylabel('Số sessions')
savefig('39_session_time_dist.png')

# 40 – Số trang đọc mỗi session
fig, ax = plt.subplots(figsize=(8,5))
ax.hist(sessions['so_trang_da_doc'], bins=12, color=PEACH,
        edgecolor='white', linewidth=0.8, zorder=3)
ax.set_title('Phân Phối Số Trang Đọc Mỗi Session', pad=12)
ax.set_xlabel('Số trang'); ax.set_ylabel('Số sessions')
savefig('40_session_pages_dist.png')

# 41 – Top recommended books
try:
    rec = pd.read_csv(os.path.join(BASE, '../output/recommendations/recommendations.csv'))

    top_books = rec['ten_sach'].value_counts().head(10)

    fig, ax = plt.subplots(figsize=(10,5))
    ax.barh(top_books.index, top_books.values, color=TEAL)
    ax.set_title("Top Recommended Books")
    ax.invert_yaxis()

    savefig('41_top_recommended_books.png')
except:
    print("⚠️ No recommendation data")

# ════════════════════════════════════════════════════════════
print("\n" + "═"*55)
charts = sorted(os.listdir('charts'))
print(f"  Tổng cộng: {len(charts)} biểu đồ")
for c in charts:
    print(f"    charts/{c}")
print("═"*55)
print("  XONG! Tất cả biểu đồ đã được lưu vào thư mục charts/")
print("═"*55)