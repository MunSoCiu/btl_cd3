import os

from src.load_data import load_data
from src.preprocess import preprocess, augment_users_realistic
from src.build_dataset import build_dataset

from src.save_charts import plot_all
from src.save_recommendations import save_recommendations
from src.save_data import save_data
from src.generate_report import generate_report
from src.time_series import run_time_series
from src.kpi import compute_kpi
from src.preprocess import preprocess, augment_users_realistic, augment_books


def main():
    print("🚀 START PIPELINE...\n")

    # ===== LOAD =====
    users, books, history = load_data()

    # ===== PREPROCESS =====
    
    users, books, history = preprocess(users, books, history)
    
    books = augment_books(books, 100)
       
    users, history = augment_users_realistic(users, books , history )

    # ===== BUILD DATASET =====
    df = build_dataset(users, books, history)

    print("✅ DATA READY:", df.shape)

    # ===== TIME SERIES =====
    ts_results, ts_metrics = run_time_series(history)
    
    if ts_results is None:
        print("⚠️ Skip time series")

    # ===== SAVE DATA =====
    save_data(df)

    # ===== CHARTS =====
    print("📊 Generating full charts...")
    os.system("python src/generate_charts.py")
    
    # ===== KPI =====
    kpi = compute_kpi(df)
    print("\n📊 KPI:")
    for k,v in kpi.items():
        print(f"{k}: {v}")

    # ===== RECOMMEND =====
    save_recommendations(users, books, history)

    # ===== REPORT =====
    generate_report(df)

    print("\n🎉 PIPELINE DONE!")
    print("\n📁 Output saved at: output/")


if __name__ == "__main__":
    main()