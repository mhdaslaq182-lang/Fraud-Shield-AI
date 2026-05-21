import numpy as np
import pandas as pd
import os

SL_BANKS = [
    "Bank of Ceylon (BOC)",
    "Peoples Bank",
    "Commercial Bank",
    "Sampath Bank",
    "HNB (Hatton National Bank)",
    "NSB (National Savings Bank)",
    "Seylan Bank",
    "NTB (Nations Trust Bank)",
    "DFCC Bank",
    "Pan Asia Bank",
    "Amana Bank",
    "Union Bank",
    "Cargills Bank",
    "MCB Bank",
    "Standard Chartered LK",
]

SL_MERCHANTS = [
    "Daraz.lk", "PickMe Food", "Dialog Axiata",
    "Hutch Lanka", "Keells Super", "Cargills Food City",
    "Odel", "Abans", "Singer Sri Lanka", "Lanka Bell",
    "SLT-Mobitel", "Dominos Pizza LK", "McDonalds LK",
    "Pizza Hut LK", "KFC Sri Lanka", "Burger King LK",
    "Softlogic", "Damro", "Nolimit", "Fashion Bug",
    "Laugfs Gas", "Litro Gas", "Ceylon Petroleum",
    "Uber Eats LK", "PickMe", "Yamu.lk",
    "Spar Lanka", "Arpico", "Laughs", "Cinnamon Hotels",
]

# ALL major cities & towns of Sri Lanka
SL_CITIES = [
    # Western Province
    "Colombo", "Dehiwala", "Sri Jayawardenepura Kotte",
    "Moratuwa", "Negombo", "Kalutara", "Panadura",
    "Homagama", "Kaduwela", "Kolonnawa", "Kelaniya",
    "Gampaha", "Ja-Ela", "Wattala", "Ragama",
    "Minuwangoda", "Katunayake", "Divulapitiya",

    # Central Province
    "Kandy", "Matale", "Nuwara Eliya", "Dambulla",
    "Hatton", "Nawalapitiya", "Peradeniya", "Gampola",
    "Teldeniya", "Kadugannawa", "Wattegama",

    # Southern Province
    "Galle", "Matara", "Hambantota", "Tangalle",
    "Ambalangoda", "Hikkaduwa", "Weligama", "Mirissa",
    "Tissamaharama", "Deniyaya", "Akuressa",

    # Northern Province
    "Jaffna", "Vavuniya", "Mannar", "Kilinochchi",
    "Mullaitivu", "Point Pedro", "Chavakachcheri",
    "Nallur", "Kopay",

    # Eastern Province
    "Trincomalee", "Batticaloa", "Ampara", "Kalmunai",
    "Akkaraipattu", "Kattankudy", "Valaichchenai",
    "Chenkalady", "Oddamavadi",

    # North Western Province
    "Kurunegala", "Puttalam", "Kuliyapitiya",
    "Wariyapola", "Maho", "Chilaw", "Wennappuwa",
    "Narammala", "Dankotuwa",

    # North Central Province
    "Anuradhapura", "Polonnaruwa", "Medawachchiya",
    "Kekirawa", "Tambuttegama", "Eppawala",
    "Mihintale", "Habarana",

    # Uva Province
    "Badulla", "Monaragala", "Bandarawela", "Haputale",
    "Welimada", "Mahiyanganaya", "Bibile", "Buttala",
    "Ella", "Diyatalawa",

    # Sabaragamuwa Province
    "Ratnapura", "Kegalle", "Balangoda", "Embilipitiya",
    "Avissawella", "Pelmadulla", "Eheliyagoda",
    "Kuruwita", "Kalawana",
]

def generate_banking_data(n_samples=10000, fraud_ratio=0.08, seed=42):
    np.random.seed(seed)

    n_fraud = int(n_samples * fraud_ratio)
    n_legit = n_samples - n_fraud
    n_each  = n_fraud // 3

    legit = pd.DataFrame({
        "amount_lkr":    np.random.exponential(8000, n_legit),
        "hour":          np.random.randint(8, 20, n_legit),
        "frequency":     np.random.poisson(4, n_legit),
        "distance_km":   np.random.exponential(10, n_legit),
        "failed_logins": np.random.choice([0,1], n_legit, p=[0.97,0.03]),
        "new_device":    np.random.choice([0,1], n_legit, p=[0.92,0.08]),
        "account_age":   np.random.randint(180, 3000, n_legit),
        "countries":     np.ones(n_legit, dtype=int),
        "velocity_24h":  np.random.poisson(2, n_legit),
        "email_risk":    np.random.uniform(0.0, 0.2, n_legit),
        "balance_lkr":   np.random.uniform(50000, 5000000, n_legit),
        "is_weekend":    np.random.choice([0,1], n_legit, p=[0.7,0.3]),
        "bank":          np.random.choice(SL_BANKS, n_legit),
        "merchant":      np.random.choice(SL_MERCHANTS, n_legit),
        "city":          np.random.choice(SL_CITIES, n_legit),
        "fraud_type":    0,
    })

    cc_fraud = pd.DataFrame({
        "amount_lkr":    np.random.uniform(50000, 500000, n_each),
        "hour":          np.random.choice([0,1,2,3,4,23], n_each),
        "frequency":     np.random.poisson(18, n_each),
        "distance_km":   np.random.uniform(300, 2000, n_each),
        "failed_logins": np.random.randint(0, 3, n_each),
        "new_device":    np.random.choice([0,1], n_each, p=[0.2,0.8]),
        "account_age":   np.random.randint(1, 45, n_each),
        "countries":     np.random.randint(2, 5, n_each),
        "velocity_24h":  np.random.poisson(12, n_each),
        "email_risk":    np.random.uniform(0.3, 0.7, n_each),
        "balance_lkr":   np.random.uniform(10000, 200000, n_each),
        "is_weekend":    np.random.choice([0,1], n_each, p=[0.5,0.5]),
        "bank":          np.random.choice(SL_BANKS, n_each),
        "merchant":      np.random.choice(SL_MERCHANTS, n_each),
        "city":          np.random.choice(SL_CITIES, n_each),
        "fraud_type":    1,
    })

    aml = pd.DataFrame({
        "amount_lkr":    np.random.choice([99900, 499900, 999900], n_each),
        "hour":          np.random.randint(0, 24, n_each),
        "frequency":     np.random.poisson(25, n_each),
        "distance_km":   np.random.uniform(100, 500, n_each),
        "failed_logins": np.zeros(n_each, dtype=int),
        "new_device":    np.zeros(n_each, dtype=int),
        "account_age":   np.random.randint(30, 200, n_each),
        "countries":     np.random.randint(3, 6, n_each),
        "velocity_24h":  np.random.poisson(20, n_each),
        "email_risk":    np.random.uniform(0.1, 0.4, n_each),
        "balance_lkr":   np.random.uniform(1000000, 10000000, n_each),
        "is_weekend":    np.random.choice([0,1], n_each, p=[0.5,0.5]),
        "bank":          np.random.choice(SL_BANKS, n_each),
        "merchant":      np.random.choice(SL_MERCHANTS, n_each),
        "city":          np.random.choice(SL_CITIES, n_each),
        "fraud_type":    2,
    })

    n_ato = n_fraud - 2 * n_each
    ato = pd.DataFrame({
        "amount_lkr":    np.random.uniform(20000, 300000, n_ato),
        "hour":          np.random.choice([1,2,3,4], n_ato),
        "frequency":     np.random.poisson(10, n_ato),
        "distance_km":   np.random.uniform(500, 3000, n_ato),
        "failed_logins": np.random.randint(5, 15, n_ato),
        "new_device":    np.ones(n_ato, dtype=int),
        "account_age":   np.random.randint(365, 2000, n_ato),
        "countries":     np.random.randint(2, 4, n_ato),
        "velocity_24h":  np.random.poisson(15, n_ato),
        "email_risk":    np.random.uniform(0.6, 1.0, n_ato),
        "balance_lkr":   np.random.uniform(100000, 2000000, n_ato),
        "is_weekend":    np.random.choice([0,1], n_ato, p=[0.5,0.5]),
        "bank":          np.random.choice(SL_BANKS, n_ato),
        "merchant":      np.random.choice(SL_MERCHANTS, n_ato),
        "city":          np.random.choice(SL_CITIES, n_ato),
        "fraud_type":    3,
    })

    df = pd.concat([legit, cc_fraud, aml, ato], ignore_index=True)
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    df["amount_lkr"]  = df["amount_lkr"].clip(100, 1000000).round(2)
    df["balance_lkr"] = df["balance_lkr"].round(2)
    df["is_fraud"]    = (df["fraud_type"] > 0).astype(int)

    os.makedirs("data", exist_ok=True)
    df.to_csv("data/banking_dataset.csv", index=False)

    print("=" * 52)
    print("  🇱🇰  SRI LANKA BANKING DATASET GENERATED")
    print("=" * 52)
    print(f"  Total transactions : {len(df):,}")
    print(f"  Legitimate         : {(df.is_fraud==0).sum():,}")
    print(f"  Fraudulent         : {(df.is_fraud==1).sum():,}")
    print(f"  Fraud rate         : {df.is_fraud.mean():.1%}")
    print(f"  Credit Card Fraud  : {(df.fraud_type==1).sum():,}")
    print(f"  Money Laundering   : {(df.fraud_type==2).sum():,}")
    print(f"  Account Takeover   : {(df.fraud_type==3).sum():,}")
    print(f"  Currency           : LKR (Sri Lankan Rupee)")
    print(f"  Banks              : {len(SL_BANKS)} Sri Lankan banks")
    print(f"  Cities             : {len(SL_CITIES)} Sri Lankan cities")
    print(f"  Merchants          : {len(SL_MERCHANTS)} local merchants")
    print(f"  Saved to           : data/banking_dataset.csv")
    print("=" * 52)
    return df


if __name__ == "__main__":
    df = generate_banking_data()
    print("\nSample transactions:")
    print(df[["amount_lkr","bank","merchant","city","fraud_type"]].head(10).to_string())
    print(f"\nAll cities covered: {len(SL_CITIES)}")
    print("Provinces: Western, Central, Southern, Northern,")
    print("           Eastern, North Western, North Central,")
    print("           Uva, Sabaragamuwa")
