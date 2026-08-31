"""
Step 2: Deep dive — Does delivery delay hurt customer satisfaction?

Business question: if late deliveries measurably hurt review scores, that's
a quantifiable case for investing in logistics/fulfillment improvements.

Run: python src/deep_dive_delivery.py
Outputs: PNG charts saved to images/, printed summary stats
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

sns.set_theme(style="whitegrid", palette="deep")
plt.rcParams["figure.dpi"] = 110

DATA_PATH = "data/orders_clean.csv"
IMG_DIR = "images"


def main():
    df = pd.read_csv(DATA_PATH)
    df = df.dropna(subset=["review_score"]).copy()

    # --- 1. Review score by on-time vs late ---
    on_time = df.loc[df["is_late"] == 0, "review_score"]
    late = df.loc[df["is_late"] == 1, "review_score"]

    print("=" * 60)
    print(f"On-time orders: {len(on_time):,}  |  avg review score: {on_time.mean():.2f}")
    print(f"Late orders:     {len(late):,}  |  avg review score: {late.mean():.2f}")

    t_stat, p_value = stats.ttest_ind(on_time, late, equal_var=False)
    print(f"\nWelch's t-test: t = {t_stat:.2f}, p = {p_value:.6f}")
    if p_value < 0.05:
        print("-> Statistically significant difference at the 5% level.")
    else:
        print("-> No statistically significant difference at the 5% level.")
    print("=" * 60)

    fig, ax = plt.subplots(figsize=(7, 5))
    sns.boxplot(
        x=df["is_late"].map({0: "On-time", 1: "Late"}),
        y=df["review_score"],
        ax=ax,
        hue=df["is_late"].map({0: "On-time", 1: "Late"}),
        legend=False,
        palette={"On-time": "#55A868", "Late": "#C44E52"},
    )
    ax.set_title("Review Score: On-time vs Late Deliveries")
    ax.set_xlabel("")
    ax.set_ylabel("Review Score (1-5)")
    fig.tight_layout()
    fig.savefig(f"{IMG_DIR}/07_review_by_lateness.png")
    plt.close(fig)

    # --- 2. Review score vs delivery days (continuous) ---
    avg_by_days = df.groupby("delivery_days")["review_score"].agg(["mean", "count"])
    avg_by_days = avg_by_days[avg_by_days["count"] >= 20]  # drop noisy low-count tail

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.scatter(df["delivery_days"], df["review_score"], alpha=0.05, color="#4C72B0")
    ax.plot(avg_by_days.index, avg_by_days["mean"], color="#C44E52", marker="o", linewidth=2,
            label="Average score per delivery-day bucket")
    ax.set_title("Review Score vs. Delivery Time")
    ax.set_xlabel("Delivery Days")
    ax.set_ylabel("Review Score")
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{IMG_DIR}/08_review_vs_delivery_days.png")
    plt.close(fig)

    corr = df["delivery_days"].corr(df["review_score"])
    print(f"\nCorrelation (delivery_days vs review_score): {corr:.3f}")

    # --- 3. Late delivery rate by state (where is this worth fixing first?) ---
    late_rate_by_state = df.groupby("customer_state")["is_late"].mean().sort_values(ascending=False) * 100
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(x=late_rate_by_state.values, y=late_rate_by_state.index, ax=ax, color="#DD8452")
    ax.set_title("Late Delivery Rate by State")
    ax.set_xlabel("% of Orders Delivered Late")
    ax.set_ylabel("")
    fig.tight_layout()
    fig.savefig(f"{IMG_DIR}/09_late_rate_by_state.png")
    plt.close(fig)

    print("\nStates with highest late-delivery rates:")
    print(late_rate_by_state.head(3).round(1))
    print("\nCharts saved to images/")


if __name__ == "__main__":
    main()
