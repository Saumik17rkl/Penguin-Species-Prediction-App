import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import streamlit as st

# Load the dataset
penguins = pd.read_csv('penguins_cleaned.csv')

# Copy dataset for modification
df = penguins.copy()

# Encode categorical features
target = 'species'
encode = ['sex', 'island']

for col in encode:
    dummies = pd.get_dummies(df[col], prefix=col, dtype=int)
    df = pd.concat([df, dummies], axis=1)
    del df[col]

# Encode the target variable
target_mapper = {
    'Adelie': 0, 'Chinstrap': 1, 'Gentoo': 2
}

df['species'] = df['species'].apply(lambda x: target_mapper[x])

# Make all column names lowercase to ensure consistency
df.columns = df.columns.str.lower()

# Define features and target
X = df.drop('species', axis=1)
Y = df['species']

# Split data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42, stratify=Y)

# Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train RandomForestClassifier
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, Y_train)

# Streamlit app
st.set_page_config(page_title="Penguin Species Predictor", layout="wide")
st.title("🐧 Penguin Species Prediction App")

st.sidebar.header("User Input Features")
st.sidebar.write("""
### Adjust the input parameters to predict the penguin species.
""")

# Sidebar inputs for user data
island = st.sidebar.selectbox("Island", ['Biscoe', 'Dream', 'Torgersen'])
sex = st.sidebar.selectbox("Sex", ['Male', 'Female'])
bill_length_mm = st.sidebar.slider("Bill Length (mm)", float(df['bill_length_mm'].min()), float(df['bill_length_mm'].max()), float(df['bill_length_mm'].mean()))
bill_depth_mm = st.sidebar.slider("Bill Depth (mm)", float(df['bill_depth_mm'].min()), float(df['bill_depth_mm'].max()), float(df['bill_depth_mm'].mean()))
flipper_length_mm = st.sidebar.slider("Flipper Length (mm)", float(df['flipper_length_mm'].min()), float(df['flipper_length_mm'].max()), float(df['flipper_length_mm'].mean()))
body_mass_g = st.sidebar.slider("Body Mass (g)", float(df['body_mass_g'].min()), float(df['body_mass_g'].max()), float(df['body_mass_g'].mean()))

# Map user input to model input format
user_data = {
    'bill_length_mm': bill_length_mm,
    'bill_depth_mm': bill_depth_mm,
    'flipper_length_mm': flipper_length_mm,
    'body_mass_g': body_mass_g,
    'sex_female': 1 if sex == 'Female' else 0,
    'sex_male': 1 if sex == 'Male' else 0,
    'island_biscoe': 1 if island == 'Biscoe' else 0,
    'island_dream': 1 if island == 'Dream' else 0,
    'island_torgersen': 1 if island == 'Torgersen' else 0,
}

user_df = pd.DataFrame(user_data, index=[0])
user_df.columns = user_df.columns.str.lower()

# Scale user input
user_scaled = scaler.transform(user_df)

# Prediction
prediction = clf.predict(user_scaled)
prediction_proba = clf.predict_proba(user_scaled)

# Map prediction to species name
species_mapper = {v: k for k, v in target_mapper.items()}
predicted_species = species_mapper[prediction[0]]

# Display prediction
st.subheader("Prediction Result")
st.markdown(f"### The predicted species is: **{predicted_species}** 🐧")

# Display prediction probabilities
st.subheader("Prediction Probabilities")
prob_df = pd.DataFrame(prediction_proba, columns=[species_mapper[v] for v in range(len(species_mapper))])
st.bar_chart(prob_df)

# Data visualization section
st.subheader("Data Insights")
st.write("### Feature Distribution")
col1, col2 = st.columns(2)

# Distribution plots
with col1:
    st.write("#### Distribution of Bill Length")
    fig, ax = plt.subplots()
    sns.histplot(df['bill_length_mm'], kde=True, ax=ax, color='blue')
    st.pyplot(fig)

with col2:
    st.write("#### Distribution of Flipper Length")
    fig, ax = plt.subplots()
    sns.histplot(df['flipper_length_mm'], kde=True, ax=ax, color='green')
    st.pyplot(fig)

st.write("### Feature Importance")
feature_importances = pd.DataFrame({'Feature': X.columns, 'Importance': clf.feature_importances_})
feature_importances = feature_importances.sort_values(by='Importance', ascending=False)

fig, ax = plt.subplots()
sns.barplot(data=feature_importances, x='Importance', y='Feature', palette='viridis', ax=ax)
st.pyplot(fig)

# Display raw data
st.subheader("Dataset")
st.write("Below is a preview of the dataset used for training the model:")
st.dataframe(penguins)
