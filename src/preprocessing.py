import pandas as pd
from sklearn.model_selection import train_test_split

RANDOM_SEED = 42

def load_and_split(csv_path):
    df = pd.read_csv(csv_path)

    # First split: 70% train, 30% temp (which becomes val+test)
    train_df, temp_df = train_test_split(
        df,
        test_size=0.30,
        stratify=df["Best Match"],
        random_state=RANDOM_SEED
    )

    # Second split: split temp 50/50 -> 15% val, 15% test overall
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        stratify=temp_df["Best Match"],
        random_state=RANDOM_SEED
    )

    return train_df, val_df, test_df

if __name__ == "__main__":
    train_df, val_df, test_df = load_and_split("data/raw/job_applicant_dataset.csv")

    print("Train shape:", train_df.shape)
    print("Val shape:", val_df.shape)
    print("Test shape:", test_df.shape)

    print("\nTrain target distribution:")
    print(train_df["Best Match"].value_counts(normalize=True))

    print("\nVal target distribution:")
    print(val_df["Best Match"].value_counts(normalize=True))

    print("\nTest target distribution:")
    print(test_df["Best Match"].value_counts(normalize=True))

    train_df.to_csv("data/processed/train.csv", index=False)
    val_df.to_csv("data/processed/val.csv", index=False)
    test_df.to_csv("data/processed/test.csv", index=False)
