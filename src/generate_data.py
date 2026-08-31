"""
Generates a synthetic e-commerce dataset that mirrors the structure of the
real Olist Brazilian E-Commerce dataset (https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce).

Why synthetic data?
This lets the whole pipeline run out-of-the-box without a Kaggle download.
To use the REAL dataset instead: download the Olist CSVs into data/raw/,
then point src/eda.py at those files instead — the column names below are
kept identical so the rest of the code needs no changes.

Run: python src/generate_data.py
"""

import numpy as np
import pandas as pd
from datetime import timedelta

np.random.seed(42)

N_ORDERS = 12000
START_DATE = pd.Timestamp("2023-01-01")
END_DATE = pd.Timestamp("2024-12-31")

STATES = ["SP", "RJ", "MG", "RS", "PR", "SC", "BA", "GO", "PE", "CE"]
STATE_WEIGHTS = [0.30, 0.15, 0.12, 0.09, 0.08, 0.07, 0.06, 0.05, 0.04, 0.04]

CATEGORIES = [
    "electronics", "furniture", "housewares", "sports_leisure",
    "beauty_health", "toys", "fashion", "computers_accessories",
    "garden_tools", "books",
]
# base price ranges per category (min, max) in BRL-like currency units
CATEGORY_PRICE = {
    "electronics": (150, 2500), "furniture": (200, 3000),
    "housewares": (20, 300), "sports_leisure": (30, 600),
    "beauty_health": (15, 250), "toys": (20, 200),
    "fashion": (25, 400), "computers_accessories": (50, 1800),
    "garden_tools": (20, 350), "books": (15, 120),
}
# relative popularity (used as sampling weights)
CATEGORY_WEIGHTS = [0.14, 0.08, 0.12, 0.11, 0.13, 0.09, 0.13, 0.10, 0.05, 0.05]


def random_dates(start, end, n):
    """Uniform random timestamps between start and end, with mild seasonality
    (more orders in Nov/Dec, fewer in Feb) and a weekday bias."""
    days_range = (end - start).days
    raw = start + pd.to_timedelta(np.random.randint(0, days_range, n), unit="D")
    df_dates = pd.Series(raw)

    month_boost = df_dates.dt.month.map({11: 1.6, 12: 1.8, 2: 0.6}).fillna(1.0)
    weekday_boost = df_dates.dt.dayofweek.map({5: 0.7, 6: 0.7}).fillna(1.1)
    keep_prob = (month_boost * weekday_boost) / 2.0
    keep_prob = keep_prob.clip(upper=1.0)
    mask = np.random.rand(n) < keep_prob.values

    # top up any dropped rows with fresh random dates until we hit n
    dates = df_dates[mask].tolist()
    while len(dates) < n:
        extra_n = n - len(dates)
        extra = start + pd.to_timedelta(np.random.randint(0, days_range, extra_n), unit="D")
        dates.extend(extra.tolist())
    return pd.to_datetime(dates[:n])


def main():
    n = N_ORDERS

    order_id = [f"ord_{i:06d}" for i in range(n)]
    customer_id = [f"cust_{i:06d}" for i in np.random.randint(0, int(n * 0.75), n)]  # some repeat customers
    order_purchase_timestamp = random_dates(START_DATE, END_DATE, n)

    customer_state = np.random.choice(STATES, size=n, p=STATE_WEIGHTS)
    product_category = np.random.choice(CATEGORIES, size=n, p=CATEGORY_WEIGHTS)

    price = np.array([
        np.random.uniform(*CATEGORY_PRICE[cat]) for cat in product_category
    ])
    # add a mild upward price trend over time + noise
    days_since_start = (order_purchase_timestamp - START_DATE).days
    trend_factor = 1 + (days_since_start / days_range_const()) * 0.15
    price = price * trend_factor
    price = np.round(price, 2)

    freight_value = np.round(price * np.random.uniform(0.03, 0.15, n), 2)

    # delivery: promised vs actual, with occasional delays
    processing_days = np.random.randint(2, 10, n)
    delay_noise = np.random.choice([0, 0, 0, 1, 2, 3, 7], size=n, p=[0.55, 0.15, 0.1, 0.08, 0.05, 0.04, 0.03])
    delivery_days = processing_days + delay_noise
    order_delivered_customer_date = order_purchase_timestamp + pd.to_timedelta(delivery_days, unit="D")
    order_estimated_delivery_date = order_purchase_timestamp + pd.to_timedelta(processing_days + 5, unit="D")

    is_late = (order_delivered_customer_date > order_estimated_delivery_date).astype(int)

    # review score: correlated with lateness
    base_score = np.random.choice([5, 4, 3, 2, 1], size=n, p=[0.45, 0.28, 0.14, 0.08, 0.05])
    review_score = np.where(is_late == 1, np.clip(base_score - np.random.choice([1, 2], n), 1, 5), base_score)

    payment_type = np.random.choice(
        ["credit_card", "boleto", "voucher", "debit_card"], size=n, p=[0.73, 0.19, 0.05, 0.03]
    )
    payment_installments = np.where(
        payment_type == "credit_card", np.random.randint(1, 10, n), 1
    )

    orders = pd.DataFrame({
        "order_id": order_id,
        "customer_id": customer_id,
        "customer_state": customer_state,
        "product_category": product_category,
        "price": price,
        "freight_value": freight_value,
        "payment_type": payment_type,
        "payment_installments": payment_installments,
        "order_purchase_timestamp": order_purchase_timestamp,
        "order_delivered_customer_date": order_delivered_customer_date,
        "order_estimated_delivery_date": order_estimated_delivery_date,
        "delivery_days": delivery_days,
        "is_late": is_late,
        "review_score": review_score,
    })

    orders = orders.sort_values("order_purchase_timestamp").reset_index(drop=True)

    # inject a small amount of realistic messiness for the cleaning step
    dupe_sample = orders.sample(frac=0.01, random_state=1)
    orders = pd.concat([orders, dupe_sample], ignore_index=True)
    missing_idx = orders.sample(frac=0.02, random_state=2).index
    orders.loc[missing_idx, "review_score"] = np.nan

    orders.to_csv("data/orders.csv", index=False)
    print(f"Wrote data/orders.csv with {len(orders):,} rows")
    print(orders.head())


def days_range_const():
    return (END_DATE - START_DATE).days


if __name__ == "__main__":
    main()
