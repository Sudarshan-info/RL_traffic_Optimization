import numpy as np
import pandas as pd
from pathlib import Path

np.random.seed(42)


def get_traffic_multiplier(hour: int) -> float:
    """Traffic intensity multiplier based on hour of day."""
    if 7 <= hour <= 9:
        return np.random.uniform(1.8, 2.5)  # Morning peak
    elif 17 <= hour <= 19:
        return np.random.uniform(2.0, 3.0)  # Evening peak
    elif 11 <= hour <= 14:
        return np.random.uniform(1.2, 1.6)  # Lunch
    elif 0 <= hour <= 5:
        return np.random.uniform(0.1, 0.3)  # Night
    else:
        return np.random.uniform(0.6, 1.2)  # Normal


def generate_data(
    n_days=30, n_intersections=4, output_path="data/traffic_data.csv"
) -> pd.DataFrame:
    records = []
    base_arrival = 8

    for day in range(n_days):
        is_weekend = day % 7 >= 5
        weekend_factor = 0.6 if is_weekend else 1.0
        for hour in range(24):
            for minute in range(0, 60, 5):
                for iid in range(n_intersections):
                    mult = get_traffic_multiplier(hour)
                    ifact = np.random.uniform(0.7, 1.3)
                    arr = max(
                        0,
                        base_arrival * mult * weekend_factor * ifact
                        + np.random.normal(0, 0.5),
                    )
                    green = np.random.randint(10, 61)
                    thru = green * 0.5
                    queue = max(0, arr * 5 - thru + np.random.normal(0, 1))
                    wait = max(0, (queue / max(1, thru)) * 30 + np.random.normal(0, 2))
                    records.append(
                        {
                            "day": day,
                            "hour": hour,
                            "minute": minute,
                            "intersection_id": iid,
                            "is_weekend": int(is_weekend),
                            "arrival_rate": round(arr, 2),
                            "green_time": green,
                            "queue_length": round(queue, 2),
                            "waiting_time": round(wait, 2),
                            "time_of_day": f"{hour:02d}:{minute:02d}",
                        }
                    )

    df = pd.DataFrame(records)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Generated {len(df):,} rows -> {output_path}")
    return df


if __name__ == "__main__":
    df = generate_data()
    print(df.head())
    print(df.describe())
