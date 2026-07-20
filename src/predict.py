import joblib
import pandas as pd

# Load trained model
model = joblib.load("models/titanic_model.pkl")

# Example passenger
passenger = pd.DataFrame(
    [
        {
            "Pclass": 3,
            "Sex": 1,        # Male
            "Age": 22,
            "SibSp": 1,
            "Parch": 0,
            "Fare": 7.25,
            "Embarked": 2,   # Southampton (encoded)
        }
    ]
)

prediction = model.predict(passenger)[0]

print("Prediction:", prediction)

if prediction == 1:
    print("Prediction : Passenger Survived")
else:
    print("Prediction : Passenger Not Survived")