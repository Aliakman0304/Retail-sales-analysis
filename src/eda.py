"""
Step 1: Exploratory Data Analysis on the retail orders dataset.

Covers:
- Data quality check (missing values, duplicates)
- Revenue trend over time
- Seasonality (day-of-week, month)
- Top product categories
- Regional performance
- Order value distribution

Run: python src/eda.py
Outputs: PNG charts saved to images/
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid", palette="deep")
plt.rcParams["figure.dpi"] = 110

DATA_PATH = "data/orders.csv"
IMG_DIR = "images"


def load_and_clean():
    df = pd.read_csv(DATA_PATH, parse_dates=[
        "order_purchase_timestamp", "order_delivered_customer_date", "order_estimated_delivery_date"
    ])

    print("=" * 60)
    print("RAW SHAPE:", df.shape)
    print("\nMissing values per column:")
    print(df.isnull().sum()[df.isnull().sum() > 0])

    n_dupes = df.duplicated(subset="order_id").sum()
    print(f"\nDuplicate order_ids: {n_dupes}")
    df = df.drop_duplicates(subset="order_id").copy()

    # review_score missing -> not something we should guess, but for revenue/volume
    # analysis we don't need it, so we leave NaNs and only drop them for review-specific plots
    df["revenue"] = df["price"] + df["freight_value"]
    df["order_month"] = df["order_purchase_timestamp"].dt.to_period("M").dt.to_timestamp()
    df["order_weekday"] = df["order_purchase_timestamp"].dt.day_name()
    df["order_year"] = df["order_purchase_timestamp"].dt.year

    print("\nCLEANED SHAPE:", df.shape)
    print("=" * 60)
    return df


def plot_missing_values(df_raw):
    fig, ax = plt.subplots(figsize=(8, 4))
    missing = df_raw.isnull().mean().sort_values(ascending=False)
    missing = missing[missing > 0]
    if len(missing) == 0:
        plt.close(fig)
        return
    sns.barplot(x=missing.values * 100, y=missing.index, ax=ax, color="#4C72B0")
    ax.set_xlabel("% missing")
    ax.set_title("Missing Values by Column")
    fig.tight_layout()
    fig.savefig(f"{IMG_DIR}/01_missing_values.png")
    plt.close(fig)


def plot_revenue_trend(df):
    monthly = df.groupby("order_month")["revenue"].sum().reset_index()
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(monthly["order_month"], monthly["revenue"], marker="o", color="#4C72B0")
    ax.set_title("Monthly Revenue Trend")
    ax.set_xlabel("Month")
    ax.set_ylabel("Revenue")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(f"{IMG_DIR}/02_monthly_revenue_trend.png")
    plt.close(fig)


def plot_seasonality(df):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    by_weekday = df.groupby("order_weekday")["revenue"].sum().reindex(weekday_order)
    sns.barplot(x=by_weekday.index, y=by_weekday.values, ax=axes[0], color="#55A868")
    axes[0].set_title("Revenue by Day of Week")
    axes[0].set_xlabel("")
    axes[0].set_ylabel("Revenue")
    axes[0].tick_params(axis="x", rotation=45)

    df["month_num"] = df["order_purchase_timestamp"].dt.month
    by_month = df.groupby("month_num")["revenue"].sum()
    sns.barplot(x=by_month.index, y=by_month.values, ax=axes[1], color="#C44E52")
    axes[1].set_title("Revenue by Month (all years combined)")
    axes[1].set_xlabel("Month")
    axes[1].set_ylabel("Revenue")

    fig.tight_layout()
    fig.savefig(f"{IMG_DIR}/03_seasonality.png")
    plt.close(fig)


def plot_top_categories(df):
    top_cat = df.groupby("product_category")["revenue"].sum().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(x=top_cat.values, y=top_cat.index, ax=ax, color="#4C72B0")
    ax.set_title("Revenue by Product Category")
    ax.set_xlabel("Revenue")
    ax.set_ylabel("")
    fig.tight_layout()
    fig.savefig(f"{IMG_DIR}/04_top_categories.png")
    plt.close(fig)
    return top_cat


def plot_regional(df):
    by_state = df.groupby("customer_state")["revenue"].sum().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(x=by_state.values, y=by_state.index, ax=ax, color="#8172B2")
    ax.set_title("Revenue by Customer State")
    ax.set_xlabel("Revenue")
    ax.set_ylabel("")
    fig.tight_layout()
    fig.savefig(f"{IMG_DIR}/05_revenue_by_state.png")
    plt.close(fig)
    return by_state


def plot_order_value_distribution(df):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    sns.histplot(df["revenue"], bins=50, ax=axes[0], color="#4C72B0")
    axes[0].set_title("Order Value Distribution")
    axes[0].set_xlabel("Revenue per Order")

    sns.boxplot(x=df["revenue"], ax=axes[1], color="#DD8452")
    axes[1].set_title("Order Value — Outlier Check")
    axes[1].set_xlabel("Revenue per Order")

    fig.tight_layout()
    fig.savefig(f"{IMG_DIR}/06_order_value_distribution.png")
    plt.close(fig)


def main():
    df_raw = pd.read_csv(DATA_PATH)
    plot_missing_values(df_raw)

    df = load_and_clean()

    plot_revenue_trend(df)
    plot_seasonality(df)
    top_cat = plot_top_categories(df)
    by_state = plot_regional(df)
    plot_order_value_distribution(df)

    print("\nTop 3 categories by revenue:")
    print(top_cat.head(3))
    print("\nTop 3 states by revenue:")
    print(by_state.head(3))
    print(f"\nTotal revenue: {df['revenue'].sum():,.2f}")
    print(f"Total orders: {df['order_id'].nunique():,}")
    print(f"Average order value: {df['revenue'].mean():,.2f}")

    df.to_csv("data/orders_clean.csv", index=False)
    print("\nSaved cleaned dataset to data/orders_clean.csv")
    print("Charts saved to images/")


if __name__ == "__main__":
    main()
