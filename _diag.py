import pandas as pd, warnings; warnings.filterwarnings('ignore')
history = pd.read_csv('data/cleaned_reading_history.csv')
books   = pd.read_csv('data/books.csv')
df_raw  = pd.read_csv('data/cleaned_dataset.csv')

STATUS_MAP = {'done':'done','Đã đọc xong':'done','reading':'reading','Đang đọc':'reading',
              'dropped':'dropped','Bỏ dở':'dropped','not_started':'not_started','Chưa bắt đầu':'not_started'}
history['trang_thai'] = history['trang_thai'].map(STATUS_MAP).fillna(history['trang_thai'])

# history already has diem_danh_gia (rating_user), books also has diem_danh_gia
# merge creates diem_danh_gia_x (history) and diem_danh_gia_y (books)
hist_full = history.merge(books[['book_id','ten_sach','the_loai','diem_danh_gia']], on='book_id', how='left')
print('hist_full columns:', list(hist_full.columns))
print('user_id nunique:', hist_full['user_id'].nunique())
print('book_id nunique:', hist_full['book_id'].nunique())
print()

# User 1
u1 = hist_full[hist_full['user_id']==1]
print('User 1 records:', len(u1))
print(u1[['book_id','ten_sach','the_loai','trang_thai']].to_string())
print()

# User 50
u50 = hist_full[hist_full['user_id']==50]
print('User 50 records:', len(u50))
print(u50[['book_id','ten_sach','the_loai','trang_thai']].to_string())
print()

# df_raw coverage
print('df_raw user_id nunique:', df_raw['user_id'].nunique())
print('df_raw user_id range:', df_raw['user_id'].min(), '-', df_raw['user_id'].max())
missing = set(history['user_id'].unique()) - set(df_raw['user_id'].unique())
print('Users in history NOT in df_raw:', len(missing))
print('Sample missing:', sorted(missing)[:10])
print()

# Time series
ts = pd.read_csv('output/data/time_series_results.csv')
print('=== TIME SERIES ===')
print('Shape:', ts.shape)
print('Columns:', list(ts.columns))
print('actual range:', ts['actual'].min(), '-', ts['actual'].max())
print('actual sum:', ts['actual'].sum())
print(ts.head(5).to_string())
