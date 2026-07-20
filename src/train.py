import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def main():
    # Load dataset
    df = pd.read_csv("data/Titanic-Dataset.csv")

    print("Dataset Shape:", df.shape)

    # Drop unnecessary columns
    df.drop(
        columns=["PassengerId", "Name", "Ticket", "Cabin"],
        inplace=True,
    )

    # Fill missing values
    df["Age"] = df["Age"].fillna(df["Age"].median())
    df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

    # Encode categorical columns
    encoder = LabelEncoder()

    for column in ["Sex", "Embarked"]:
        df[column] = encoder.fit_transform(df[column])

    # Features and target
    X = df.drop("Survived", axis=1)
    y = df["Survived"]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    # Train model
    model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
    )

    model.fit(X_train, y_train)

    # Evaluate
    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    print(f"Random Forest Accuracy: {accuracy:.4f}")

    # Save model
    joblib.dump(model, "models/titanic_model.pkl")

    print("Model saved in models/titanic_model.pkl")


if __name__ == "__main__":
    main()