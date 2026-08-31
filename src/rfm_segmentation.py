"""
Step 3: RFM Customer Segmentation
Recency, Frequency, Monetary — a standard technique for identifying
high-value customers and at-risk customers worth targeting with retention campaigns.

Run: python src/rfm_segmentation.py
Outputs: PNG chart saved to images/, data/rfm_segments.csv
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid", palette="deep")
plt.rcParams["figure.dpi"] = 110

DATA_PATH = "data/orders_clean.csv"
IMG_DIR = "images"


def main():
    df = pd.read_csv(DATA_PATH, parse_dates=["order_purchase_timestamp"])

    snapshot_date = df["order_purchase_timestamp"].max() + pd.Timedelta(days=1)

    rfm = df.groupby("customer_id").agg(
        recency=("order_purchase_timestamp", lambda x: (snapshot_date - x.max()).days),
        frequency=("order_id", "nunique"),
        monetary=("revenue", "sum"),
    ).reset_index()

    # score 1 (worst) to 4 (best) per metric using quartiles
    rfm["r_score"] = pd.qcut(rfm["recency"], 4, labels=[4, 3, 2, 1]).astype(int)
    rfm["f_score"] = pd.qcut(rfm["frequency"].rank(method="first"), 4, labels=[1, 2, 3, 4]).astype(int)
    rfm["m_score"] = pd.qcut(rfm["monetary"], 4, labels=[1, 2, 3, 4]).astype(int)
    rfm["rfm_score"] = rfm["r_score"] + rfm["f_score"] + rfm["m_score"]

    def segment(row):
        if row["rfm_score"] >= 10:
            return "Champions"
        elif row["rfm_score"] >= 8:
            return "Loyal Customers"
        elif row["rfm_score"] >= 6:
            return "Potential Loyalists"
        elif row["r_score"] <= 2 and row["m_score"] >= 3:
            return "At Risk (high value)"
        else:
            return "Needs Attention"

    rfm["segment"] = rfm.apply(segment, axis=1)

    print("=" * 60)
    print("Customer segment counts:")
    print(rfm["segment"].value_counts())
    print("\nAvg monetary value by segment:")
    print(rfm.groupby("segment")["monetary"].mean().sort_values(ascending=False).round(2))
    print("=" * 60)

    fig, ax = plt.subplots(figsize=(9, 6))
    palette = {
        "Champions": "#2E7D32", "Loyal Customers": "#55A868",
        "Potential Loyalists": "#4C72B0", "At Risk (high value)": "#C44E52",
        "Needs Attention": "#B0B0B0",
    }
    for seg, sub in rfm.groupby("segment"):
        ax.scatter(sub["frequency"], sub["monetary"], alpha=0.5, s=25,
                   label=seg, color=palette.get(seg, "#888888"))
    ax.set_xlabel("Frequency (# orders)")
    ax.set_ylabel("Monetary (total revenue)")
    ax.set_title("Customer Segments — RFM Analysis")
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    fig.savefig(f"{IMG_DIR}/10_rfm_segments.png")
    plt.close(fig)

    rfm.to_csv("data/rfm_segments.csv", index=False)
    print("\nSaved data/rfm_segments.csv and images/10_rfm_segments.png")


if __name__ == "__main__":
    main()
