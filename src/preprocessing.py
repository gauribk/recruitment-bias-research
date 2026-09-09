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

from sklearn.preprocessing import OneHotEncoder
import joblib

CATEGORICAL_COLS_FULL = ["Gender", "Race", "Ethnicity", "Job Roles"]
CATEGORICAL_COLS_NO_SENSITIVE = ["Job Roles"]  # Experiment B keeps this, drops Gender/Race/Ethnicity
NUMERIC_COLS = ["Age"]

def encode_categoricals(train_df, val_df, test_df, categorical_cols):
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    encoder.fit(train_df[categorical_cols])

    def transform(df):
        encoded = encoder.transform(df[categorical_cols])
        encoded_df = pd.DataFrame(
            encoded,
            columns=encoder.get_feature_names_out(categorical_cols),
            index=df.index
        )
        return pd.concat([df[NUMERIC_COLS].reset_index(drop=True),
                           encoded_df.reset_index(drop=True)], axis=1)

    return transform(train_df), transform(val_df), transform(test_df), encoder


if __name__ == "__main__":
    train_df, val_df, test_df = load_and_split("data/raw/job_applicant_dataset.csv")

    # Experiment A: WITH sensitive attributes
    X_train_A, X_val_A, X_test_A, encoder_A = encode_categoricals(
        train_df, val_df, test_df, CATEGORICAL_COLS_FULL
    )

    # Experiment B: WITHOUT sensitive attributes
    X_train_B, X_val_B, X_test_B, encoder_B = encode_categoricals(
        train_df, val_df, test_df, CATEGORICAL_COLS_NO_SENSITIVE
    )

    print("Experiment A (with sensitive attrs) shape:", X_train_A.shape)
    print("Experiment B (without sensitive attrs) shape:", X_train_B.shape)

    X_train_A.to_csv("data/processed/X_train_experimentA.csv", index=False)
    X_train_B.to_csv("data/processed/X_train_experimentB.csv", index=False)

    joblib.dump(encoder_A, "data/processed/encoder_experimentA.joblib")
    joblib.dump(encoder_B, "data/processed/encoder_experimentB.joblib")

from sklearn.feature_extraction.text import TfidfVectorizer

def vectorize_resumes(train_df, val_df, test_df, max_features=300):
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        stop_words="english",
        lowercase=True
    )
    vectorizer.fit(train_df["Resume"])

    def transform(df):
        matrix = vectorizer.transform(df["Resume"]).toarray()
        return pd.DataFrame(
            matrix,
            columns=[f"resume_tfidf_{w}" for w in vectorizer.get_feature_names_out()],
            index=df.index
        ).reset_index(drop=True)

    return transform(train_df), transform(val_df), transform(test_df), vectorizer


if __name__ == "__main__":
    # ... (keep everything from before, then add:)

    X_train_resume, X_val_resume, X_test_resume, resume_vectorizer = vectorize_resumes(
        train_df, val_df, test_df
    )

    print("Resume TF-IDF train shape:", X_train_resume.shape)
    print("Sample resume feature names:", list(X_train_resume.columns[:10]))

    X_train_resume.to_csv("data/processed/X_train_resume_tfidf.csv", index=False)
    joblib.dump(resume_vectorizer, "data/processed/resume_tfidf_vectorizer.joblib")
