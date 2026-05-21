import pandas as pd

def engineer_features(df):
    df = df.copy()

    df["risk_score"] = (
        df["failed_logins"]   * 0.30 +
        df["email_risk"]      * 0.25 +
        df["new_device"]      * 0.20 +
        (df["countries"] - 1) * 0.15 +
        (df["velocity_24h"] / (df["velocity_24h"].max() + 1)) * 0.10
    )

    df["amount_per_velocity"] = df["amount_lkr"] / (df["velocity_24h"] + 1)
    df["is_night"]            = (df["hour"] <= 5).astype(int)
    df["new_account"]         = (df["account_age"] < 30).astype(int)
    df["amount_to_balance"]   = df["amount_lkr"] / (df["balance_lkr"] + 1)
    df["is_round_amount"]     = df["amount_lkr"].apply(
        lambda x: 1 if x in [99900, 499900, 999900] else 0
    )
    return df


FEATURES = [
    "amount_lkr", "hour", "frequency", "distance_km",
    "failed_logins", "new_device", "account_age",
    "countries", "velocity_24h", "email_risk",
    "balance_lkr", "is_weekend",
    "risk_score", "amount_per_velocity",
    "is_night", "new_account",
    "amount_to_balance", "is_round_amount"
]


if __name__ == "__main__":
    from data_generator import generate_banking_data
    df = generate_banking_data()
    df = engineer_features(df)
    print("✅ Features ready!")
    print(f"   Total: {len(FEATURES)} features")
    print(df[["amount_lkr","bank","city","risk_score"]].head())
