import os
import pandas as pd

OUTPUT_PATH = "output"

def generate_report(df):
    os.makedirs(OUTPUT_PATH, exist_ok=True)

    report = []

    report.append("=== BOOK DATA ANALYSIS REPORT ===\n")

    report.append(f"Total records: {len(df)}\n")

    report.append("\nTop Genres:\n")
    report.append(str(df['the_loai'].value_counts().head()))

    report.append("\n\nRating Stats:\n")
    report.append(str(df['rating_book'].describe()))

    report.append("\n\nTop Users:\n")
    report.append(str(df['user_id'].value_counts().head()))
    
    report.append("\n\nTime Series Metrics:\n")

    try:
         ts = pd.read_csv("output/data/time_series_metrics.csv")
         report.append(str(ts))
    except:
         report.append("No time series data")


    with open(f"{OUTPUT_PATH}/report.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(report))

    print("✅ Report generated")