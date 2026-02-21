from pathlib import Path
import pandas as pd
import numpy as np
import ast


def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    return R * 2 * np.arcsin(np.sqrt(a))


def load_courier_workload(data_dir):
    path = data_dir / "courier_wave_info_meituan.csv"
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    df["order_count"] = df["order_ids"].apply(lambda x: len(ast.literal_eval(x)))
    workload = df.groupby("courier_id")["order_count"].mean().reset_index()
    workload.columns = ["courier_id", "avg_orders_per_wave"]
    print(f"Courier workload lastet: {len(workload)} couriere")
    return workload


def filter_invalid_waybills(input_csv, output_csv):
    print(f"Leser inn {input_csv.name}...")
    df = pd.read_csv(input_csv)
    df = df.drop(columns=["Unnamed: 0"], errors="ignore")
    df.columns = df.columns.str.strip()

    # Steg 1: Fjern ordrer som ikke ble tatt av courier
    df = df[df["is_courier_grabbed"] != 0]

    # Steg 2: Fjern rader med 0 eller NaN i kritiske tidsfelt
    for col in ["arrive_time", "estimate_arrived_time", "platform_order_time"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df[df[col].notna() & (df[col] != 0)]

    # Steg 3: Konsistenssjekk – differansen skal være 0–5 timer
    diff = df["estimate_arrived_time"] - df["platform_order_time"]
    df = df[(diff >= 0) & (diff <= 18000)]

    # Steg 4: Lag target-variabel og sjekk klasseubalanse
    df["is_late"] = (df["arrive_time"] > df["estimate_arrived_time"]).astype(int)
    late_share = df["is_late"].mean()
    print(f"Andel forsinkede ordrer: {late_share:.2%}")
    if late_share < 0.10:
        print("ADVARSEL: Stor klasseubalanse – mindre enn 10% forsinkede ordrer!")

    # Steg 5: Engineered features
    dt = pd.to_datetime(df["platform_order_time"], unit="s")
    df["hour"] = dt.dt.hour
    df["day_of_week"] = dt.dt.weekday
    df["is_peak"] = df["hour"].isin([11, 12, 13, 14, 17, 18, 19, 20]).astype(int)

    df["merchant_customer_distance_km"] = haversine(
        df["sender_lat"] / 1e6,
        df["sender_lng"] / 1e6,
        df["recipient_lat"] / 1e6,
        df["recipient_lng"] / 1e6
    )

    df["estimated_prep_time_minutes"] = (
        df["estimate_meal_prepare_time"] - df["platform_order_time"]
    ) / 60
    df.loc[
        (df["estimated_prep_time_minutes"] < 0) | (df["estimated_prep_time_minutes"] > 120),
        "estimated_prep_time_minutes"
    ] = np.nan

    df["estimated_delivery_duration_minutes"] = (
        df["estimate_arrived_time"] - df["platform_order_time"]
    ) / 60

    # Steg 6: Courier workload
    workload = load_courier_workload(input_csv.parent)
    df = df.merge(workload, on="courier_id", how="left")
    df["avg_orders_per_wave"] = df["avg_orders_per_wave"].fillna(workload["avg_orders_per_wave"].mean())

    # Steg 7: Behold kun relevante kolonner
    cols_to_keep = [
        "is_prebook", "is_weekend", "poi_id", "da_id",
        "sender_lat", "sender_lng", "recipient_lat", "recipient_lng",
        "platform_order_time", "estimate_meal_prepare_time",
        "estimate_arrived_time", "arrive_time",
        "hour", "day_of_week", "is_peak",
        "merchant_customer_distance_km",
        "estimated_prep_time_minutes",
        "estimated_delivery_duration_minutes",
        "avg_orders_per_wave",
        "is_late"
    ]
    df = df[cols_to_keep]

    df.to_csv(output_csv, index=False)
    print(f"Lagret til: {output_csv}")


if __name__ == "__main__":
    project_root = Path(__file__).parent.parent
    input_path = project_root / "data" / "all_waybill_info_meituan_0322.csv"
    output_path = input_path.parent / f"filtered_{input_path.name}"
    filter_invalid_waybills(input_path, output_path)