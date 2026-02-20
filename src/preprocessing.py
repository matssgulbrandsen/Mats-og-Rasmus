from pathlib import Path
import pandas as pd

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

    # Steg 4: Feature construction fra platform_order_time
    dt = pd.to_datetime(df["platform_order_time"], unit="s")
    df["hour"] = dt.dt.hour
    df["is_weekend"] = (dt.dt.weekday >= 5).astype(int)
    df["is_peak"] = df["hour"].isin([11, 12, 13, 17, 18, 19, 20]).astype(int)

    # Steg 5: Lag target-variabel og sjekk klasseubalanse
    df["is_late"] = (df["arrive_time"] > df["estimate_arrived_time"]).astype(int)
    late_share = df["is_late"].mean()
    print(f"Andel forsinkede ordrer: {late_share:.2%}")
    if late_share < 0.10:
        print("ADVARSEL: Stor klasseubalanse – mindre enn 10% forsinkede ordrer!")
    
    cols_to_keep = ["is_prebook","is_weekend","poi_id","da_id","sender_lat","sender_lng","recipient_lat","recipient_lng","platform_order_time","estimate_meal_prepare_time","estimate_arrived_time","arrive_time","hour","is_peak"]
    df = df[cols_to_keep]

    df.to_csv(output_csv, index=False)
    print(f"Lagret til: {output_csv}")

if __name__ == "__main__":
    project_root = Path(__file__).parent.parent
    input_path = project_root / "data" / "all_waybill_info_meituan_0322.csv"
    output_path = input_path.parent / f"filtered_{input_path.name}"
    filter_invalid_waybills(input_path, output_path)
