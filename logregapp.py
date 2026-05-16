import streamlit as st
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# --------------------------------
# PAGE CONFIG
# --------------------------------

st.set_page_config(
    page_title="Employee Job Change Prediction",
    page_icon="💼",
    layout="centered"
)

# --------------------------------
# TITLE
# --------------------------------

st.title("💼 Employee Job Change Prediction")

st.write(
    "Predict whether an employee is more likely to stay or leave the company."
)

# --------------------------------
# LOAD DATASET
# --------------------------------

df = pd.read_csv("aug_train.csv")

# select useful columns
df = df[
    [
        "city_development_index",
        "training_hours",
        "experience",
        "education_level",
        "target"
    ]
]

# remove missing values
df = df.dropna()

# --------------------------------
# CLEAN EXPERIENCE COLUMN
# --------------------------------

df["experience"] = df["experience"].astype(str)

df["experience"] = df["experience"].replace(
    ">20",
    "21"
)

df["experience"] = df["experience"].replace(
    "<1",
    "0"
)

df["experience"] = pd.to_numeric(
    df["experience"],
    errors="coerce"
)

df = df.dropna()

df["experience"] = df["experience"].astype(int)

# --------------------------------
# ENCODE EDUCATION
# --------------------------------

encoder = LabelEncoder()

df["education_level"] = encoder.fit_transform(
    df["education_level"]
)

# --------------------------------
# FEATURES & TARGET
# --------------------------------

x = df.drop("target", axis=1)

y = df["target"]

# --------------------------------
# FEATURE SCALING
# --------------------------------

scaler = StandardScaler()

x = scaler.fit_transform(x)

# --------------------------------
# SPLIT DATA
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

model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced"
)

model.fit(x_train, y_train)

# --------------------------------
# USER INPUTS
# --------------------------------

st.subheader("Enter Employee Details")

# job opportunities
city_option = st.selectbox(
    "Job Opportunities in Employee's City",
    [
        "Low",
        "Medium",
        "High"
    ]
)

# corrected mapping
if city_option == "Low":
    city_index = 0.90

elif city_option == "Medium":
    city_index = 0.60

else:
    city_index = 0.30

# skill development hours
training_hours = st.slider(
    "Skill Development Hours in a Day",
    1,
    12,
    3
)

# convert daily hours to dataset scale
training_hours = training_hours * 30

# experience
experience = st.slider(
    "Years of Experience",
    0,
    25,
    5
)

# education
education = st.selectbox(
    "Education Level",
    [
        "Graduate",
        "Masters",
        "Phd"
    ]
)

education_encoded = encoder.transform(
    [education]
)[0]

# --------------------------------
# PREDICTION
# --------------------------------

if st.button("Predict"):

    input_data = [[
        city_index,
        training_hours,
        experience,
        education_encoded
    ]]

    # scale input
    input_data = scaler.transform(input_data)

    # predict probabilities
    probability = model.predict_proba(
        input_data
    )

    stay_probability = probability[0][0]
    leave_probability = probability[0][1]

    # percentages
    stay_percent = round(
        stay_probability * 100,
        2
    )

    leave_percent = round(
        leave_probability * 100,
        2
    )

    st.subheader("Prediction Result")

    # final prediction
    if stay_probability > leave_probability:

        st.success(
            "✅ Employee is More Likely to Stay in the Company"
        )

    else:

        st.error(
            "🚨 Employee is More Likely to Leave the Company"
        )

    # show probabilities
    st.write(
        "Chance of Staying:",
        stay_percent,
        "%"
    )

    st.write(
        "Chance of Leaving:",
        leave_percent,
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