import pandas as pd

def merge_recommendations():
    rec1 = pd.read_csv("data/recommendations_new.csv")
    rec2 = pd.read_csv("data/recommendations_final.csv")

    merged = pd.concat([rec1, rec2])

    # bỏ trùng
    merged = merged.drop_duplicates()

    merged.to_csv("output/data/recommendations_merged.csv", index=False)

    print("✅ Merged recommendations saved")

    return merged