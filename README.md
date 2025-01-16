# Penguin-Species-Prediction-App
This Streamlit app predicts penguin species (Adelie, Chinstrap, Gentoo) using user inputs like island, sex, and physical traits. Powered by a Random Forest model, it shows predictions, probabilities, feature importance, and data insights via interactive visuals, offering a fun and educational experience.

Penguin Species Prediction App
This project is a machine learning web application built with Streamlit that allows users to predict the species of a penguin based on various physical features such as bill length, bill depth, flipper length, body mass, and island location. The app uses a trained classification model to predict whether a penguin belongs to one of the three species: Adelie, Chinstrap, or Gentoo.

Key Features:
Interactive Inputs:

Users can input physical measurements like bill length, bill depth, flipper length, and body mass to predict the species of a penguin.
Real-Time Prediction:

The model instantly predicts the species based on the inputted values.
Species Information:

Displays relevant details about the predicted species, including general characteristics and distribution.
Model Evaluation:

Includes metrics such as accuracy, precision, and recall to evaluate the performance of the model.
Technologies Used:
Streamlit: For building the interactive web interface.
Scikit-learn: For implementing the machine learning model and model evaluation.
Pandas: For data manipulation and preprocessing.
NumPy: For numerical operations and calculations.
Matplotlib & Seaborn: For visualizations, such as plotting feature distributions and model evaluation metrics.
Dataset:
The app uses the Palmer Penguins dataset, which includes data on three species of penguins:

Adelie
Chinstrap
Gentoo
The dataset contains features such as:

Bill length and depth (in mm)
Flipper length (in mm)
Body mass (in grams)
Island (location where the penguin was observed)
Machine Learning Model:
The model used for prediction is a Random Forest Classifier, trained on the Palmer Penguins dataset. The classifier predicts the species of a penguin based on the input features. The model has been evaluated for accuracy and is optimized for real-time predictions.

How to Run:
Clone the repository:
bash
Copy
Edit
git clone https://github.com/Saumik17rkl/Penguin-Species-Prediction-App.git
Install the required libraries:
bash
Copy
Edit
pip install -r requirements.txt
Run the Streamlit app:
bash
Copy
Edit
streamlit run app.py
Screenshots:
Home Page: The input page where users can enter penguin measurements to predict the species.
Prediction Result: Displays the predicted species and relevant details.
Model Evaluation Metrics: Shows the model’s performance metrics such as accuracy, precision, and recall.
Contributions:
Feel free to fork the repository and submit improvements. If you have suggestions, issues, or want to contribute, feel free to submit a pull request.
