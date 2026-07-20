import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

DATA_PATH = "data/Titanic-Dataset.csv"
MODEL_PATH = "models/titanic_model.pkl"


def preprocess_data(df):
    """Perform basic preprocessing on the Titanic dataset."""

    df = df.drop(columns=["PassengerId", "Name", "Ticket", "Cabin"])

    df["Age"] = df["Age"].fillna(df["Age"].median())
    df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

    encoder = LabelEncoder()
    for column in ["Sex", "Embarked"]:
        df[column] = encoder.fit_transform(df[column])

    return df


def main():
    df = pd.read_csv(DATA_PATH)
    print(f"Dataset Shape: {df.shape}")

    df = preprocess_data(df)

    X = df.drop(columns="Survived")
    y = df["Survived"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    print(f"Random Forest Accuracy: {accuracy:.4f}")

    joblib.dump(model, MODEL_PATH)
    print(f"Model saved in {MODEL_PATH}")


if __name__ == "__main__":
    main()