import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# --------------------------------
# PAGE CONFIG
# --------------------------------

st.set_page_config(
    page_title="Gaming Product Purchase Classification",
    page_icon="🎮",
    layout="centered"
)

# --------------------------------
# TITLE
# --------------------------------

st.title("🎮 Gaming Product Purchase Classification")

st.write(
    "Predict whether a customer is likely to purchase a gaming product."
)

# --------------------------------
# CREATE CUSTOM DATASET
# --------------------------------

np.random.seed(42)

data = []

# younger customers -> more purchases
for i in range(200):

    age = np.random.randint(18, 30)

    salary = np.random.randint(
        20000,
        80000
    )

    purchased = np.random.choice(
        [1, 0],
        p=[0.75, 0.25]
    )

    data.append([
        age,
        salary,
        purchased
    ])

# middle-age customers
for i in range(120):

    age = np.random.randint(30, 45)

    salary = np.random.randint(
        40000,
        120000
    )

    purchased = np.random.choice(
        [1, 0],
        p=[0.50, 0.50]
    )

    data.append([
        age,
        salary,
        purchased
    ])

# older customers -> fewer purchases
for i in range(100):

    age = np.random.randint(45, 65)

    salary = np.random.randint(
        50000,
        150000
    )

    purchased = np.random.choice(
        [1, 0],
        p=[0.20, 0.80]
    )

    data.append([
        age,
        salary,
        purchased
    ])

# dataframe
df = pd.DataFrame(
    data,
    columns=[
        "Age",
        "Salary",
        "Purchased"
    ]
)

# --------------------------------
# FEATURES & TARGET
# --------------------------------

x = df[
    [
        "Age",
        "Salary"
    ]
]

y = df["Purchased"]

# --------------------------------
# FEATURE SCALING
# --------------------------------

scaler = StandardScaler()

x = scaler.fit_transform(x)

# --------------------------------
# TRAIN TEST SPLIT
# --------------------------------

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=42
)

# --------------------------------
# MODEL
# --------------------------------

model = LogisticRegression()

model.fit(x_train, y_train)

# --------------------------------
# USER INPUTS
# --------------------------------

st.subheader("Enter Customer Details")

# age
age = st.slider(
    "Customer Age",
    min_value=18,
    max_value=65,
    value=22
)

# salary
salary = st.slider(
    "Customer Annual Salary",
    min_value=20000,
    max_value=150000,
    value=50000,
    step=5000
)

# --------------------------------
# PREDICTION
# --------------------------------

if st.button("Predict"):

    input_data = [[
        age,
        salary
    ]]

    # scale input
    input_data = scaler.transform(
        input_data
    )

    # prediction
    prediction = model.predict(
        input_data
    )

    # probability
    probability = model.predict_proba(
        input_data
    )

    buy_probability = round(
        probability[0][1] * 100,
        2
    )

    st.subheader("Prediction Result")

    # final prediction
    if prediction[0] == 1:

        st.success(
            "✅ Customer is Likely to Purchase the Gaming Product"
        )

    else:

        st.error(
            "❌ Customer is Not Likely to Purchase the Gaming Product"
        )

    # probability
    st.write(
        "Chance of Purchasing:",
        buy_probability,
        "%"
    )

# --------------------------------
# MODEL ACCURACY
# --------------------------------

y_pred = model.predict(x_test)

accuracy = accuracy_score(
    y_test,
    y_pred
)

st.write(
    "Model Accuracy:",
    round(accuracy * 100, 2),
    "%"
)