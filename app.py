import streamlit as st
import pandas as pd
import joblib

# Load model and encoders
model = joblib.load('titanic_model.pkl')
sex_encoder = joblib.load('sex_encoder.pkl')
embark_encoder = joblib.load('embark_encoder.pkl')
deck_encoder = joblib.load('deck_encoder.pkl')

st.title("Titanic Survival Predictor")

# User inputs
pclass = st.selectbox("Passenger Class", [1, 2, 3])
age = st.number_input("Age", min_value=0, max_value=100, value=25)
sibsp = st.number_input("Siblings/Spouses aboard", min_value=0, max_value=10, value=0)
parch = st.number_input("Parents/Children aboard", min_value=0, max_value=10, value=0)
fare = st.number_input("Fare", min_value=0.0, value=32.0)
sex = st.selectbox("Sex", sex_encoder.classes_)
embarked = st.selectbox("Embarked", embark_encoder.classes_)
deck = st.selectbox("Deck", deck_encoder.classes_)

if st.button("Predict"):
    sex_enc = sex_encoder.transform([sex])[0]
    embark_enc = embark_encoder.transform([embarked])[0]
    deck_enc = deck_encoder.transform([deck])[0]

    input_df = pd.DataFrame([[pclass, age, sibsp, parch, fare, sex_enc, embark_enc, deck_enc]],
                             columns=['Pclass', 'Age', 'SibSp', 'Parch', 'Fare',
                                      'sex_encoded', 'embark_encoded', 'deck_encoded'])

    prediction = model.predict(input_df)[0]
    result = "Survived ✅" if prediction > 0.5 else "Did not survive ❌"

    st.subheader(result)
    st.write(f"Raw model output: {prediction:.3f}")