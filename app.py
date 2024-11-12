import streamlit as st
import pandas as pd
import joblib
import os
from sklearn.feature_extraction.text import CountVectorizer
from PIL import Image
import hashlib

import streamlit as st


# Load the trained models
# rf_model = joblib.load("rf_model.joblib")
# catboost_model = joblib.load("catboost_model.joblib")
voting_model = joblib.load("voting_model.joblib")

# Set up vectorizer with the same k-mer setup used in training
k = 3  # Replace with the k-value used in training
vectorizer = CountVectorizer(analyzer='char', ngram_range=(k, k))
vectorizer.fit(pd.Series(["ATCG"]))  # Placeholder fit, as the vectorizer would need training data

# Define the prediction function
def predict_antimicrobial(sequence):
    # Prepare input data with realistic default values based on common training data values
    input_data = pd.DataFrame({
        'Sequence': [sequence],
        'Sequence_Length': [len(sequence)],
        'Source': ['synthetic'],  # Replace 'unknown' with typical value
        'Protein_existence': ['evidence at protein level'],
        'Target_Organism': ['Escherichia coli'],
        'Hemolytic_activity': ['no'],
        'Linear/Cyclic/Branched': ['linear'],
        'Stereochemistry': ['achiral']
    })
    print("input data: ", input_data)
    # Transform the 'Sequence' into k-mer representation
    input_kmers = vectorizer.transform(input_data['Sequence']).toarray()
    kmers_df = pd.DataFrame(input_kmers, columns=vectorizer.get_feature_names_out())

    # Combine k-mer features with other columns
    input_combined = pd.concat([kmers_df, input_data[['Sequence_Length', 'Source', 'Protein_existence',
                                                       'Target_Organism', 'Hemolytic_activity',
                                                       'Linear/Cyclic/Branched', 'Stereochemistry']]], axis=1)

    # One-hot encode categorical variables
    input_encoded = pd.get_dummies(input_combined, columns=['Source', 'Protein_existence', 'Target_Organism',
                                                            'Hemolytic_activity', 'Linear/Cyclic/Branched',
                                                            'Stereochemistry'], drop_first=True)

    # Align columns with the model’s expected feature set
    # Get expected columns from the voting model (assuming it was trained on X_resampled)
    expected_columns = voting_model.estimators_[0].feature_names_in_  # Uses feature names from the RF estimator in the voting model
    input_encoded = input_encoded.reindex(columns=expected_columns, fill_value=0)

    print("encoded input: ", input_encoded)
    # Predict with the voting model
    prediction = voting_model.predict(input_encoded)
    return prediction[0]

# File path to store user credentials
user_file = "users.csv"

# Helper functions
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def authenticate(username, password):
    if os.path.exists(user_file):
        users_df = pd.read_csv(user_file)
        hashed_password = hash_password(password)
        if any((users_df['username'] == username) & (users_df['password'] == hashed_password)):
            return True
    return False

def register_user(username, password):
    hashed_password = hash_password(password)
    new_user = pd.DataFrame({'username': [username], 'password': [hashed_password]})
    
    if os.path.exists(user_file):
        users_df = pd.read_csv(user_file)
        if username in users_df['username'].values:
            return False  # Username already exists
        users_df = pd.concat([users_df, new_user], ignore_index=True)
    else:
        users_df = new_user
        
    users_df.to_csv(user_file, index=False)
    return True

# Sidebar for navigation
page = st.sidebar.selectbox("Select Page", ["Register", "Login/Home", "Project Information"])

# Register Page
if page == "Register":
    st.title("Register")
    new_username = st.text_input("Enter a Username")
    new_password = st.text_input("Enter a Password", type="password")
    confirm_password = st.text_input("Confirm Password", type="password")

    if st.button("Register"):
        if new_password == confirm_password:
            if register_user(new_username, new_password):
                st.success("Registration successful! You can now log in from the Home page.")
            else:
                st.warning("Username already exists. Please choose a different username.")
        else:
            st.warning("Passwords do not match. Please try again.")

# About Dataset and Model Page (accessible without login)
elif page == "Project Information":
    st.title("Ensemble Classifier for Antimicrobial Peptides")
    # 1st Section: About Dataset
    
    st.subheader("About Dataset")
    st.write(
        """
        This dataset contains around 6,035 entries, each providing detailed information on various antimicrobial peptides (AMPs).
        Key fields include unique identifiers, sequences, sequence length, protein families, sources, activities, structure descriptions,
        target organisms, and modifications. The dataset also documents protein existence, hemolytic activity, cytotoxicity, and specific
        antimicrobial efficacy against pathogens. Additional metadata includes PubMed IDs and references for each entry, making it a
        valuable resource for research on AMPs, particularly lantibiotics, and their applications in fighting resistant pathogens.
        """
    )

    # 2nd Section: Steps
    st.subheader("Steps for Model Preparation")
    st.write(
        """
        **Feature Selection**
        - Selected relevant columns from the dataset for features (X) and target variable (y).

        **K-mer Representation for Sequences**
        - Used CountVectorizer to convert peptide sequences into k-mers of length 3.
        - Transformed sequences into overlapping sub-sequences to capture sequence patterns for model input.

        **Combining k-mer Features with Other Features**
        - Merged k-mer representations with other selected features to form a complete feature set (X_combined).

        **Encoding Categorical Variables**
        - Applied one-hot encoding to categorical columns (e.g., Source, Protein_existence) to convert them into numerical form.
        - Used `drop_first=True` to avoid redundancy in encoded variables.

        **Splitting the Dataset**
        - Split X_encoded and y into training and testing sets with an 80/20 train-test split.

        **Addressing Class Imbalance with SMOTE**
        - Applied Synthetic Minority Over-sampling Technique (SMOTE) on the training data to balance classes.
        - Generated synthetic samples for minority classes, improving model performance on underrepresented classes.

        **Standardization**
        - Standardized features in X_resampled and X_test using StandardScaler to ensure each feature has a mean of 0 and standard deviation of 1.
        """
    )

    # 3rd Section: About the Models
    st.subheader("About the Models")
    st.write(
        """
        **Random Forest**
        - An ensemble learning method that builds multiple decision trees on random subsets of the data, combining predictions to improve accuracy and reduce overfitting.

        **CatBoost**
        - A gradient boosting algorithm that efficiently handles categorical features and reduces overfitting, especially useful for datasets with many categorical variables.

        **Logistic Regression**
        - A binary classification algorithm that models the probability of a class using a logistic function, providing probabilistic outputs for soft voting in ensembles.

        **Support Vector Machine (SVM)**
        - A supervised learning algorithm that finds the optimal hyperplane to separate classes, using soft margins and kernel tricks for handling non-linear data separation.

        **CountVectorizer**
        - Transforms peptide sequences into structured data by tokenizing into k-mers and counting their occurrences, capturing sequence patterns for model input.

        **Voting Model**
        - Combines predictions from multiple models with soft voting, where each model provides a probability, and the average is used for final predictions, improving accuracy and balance.
        """
    )

    # 4th Section: Bayesian Optimization for Hyperparameter Tuning
    st.subheader("Bayesian Optimization for Hyperparameter Tuning")
    st.write(
        """
        The bounds in Bayesian Optimization define the search space for tuning hyperparameters of the RandomForestClassifier and CatBoostClassifier.
        - **RandomForestClassifier**: 
          - `n_estimators` between 50 and 150, `max_depth` between 5 and 15, and `min_samples_split` between 2 and 8.
        - **CatBoostClassifier**:
          - `iterations` between 100 and 500, `learning_rate` between 0.05 and 0.2, and `depth` between 3 and 8.

        These bounds limit the parameter search space, allowing Bayesian Optimization to efficiently find the best hyperparameters for improved model performance.
        """
    )

    # 5th Section: Display Images
    st.subheader("Model Analysis and Performance Visualizations")

    # Load and display images with titles
    image_files = {
        "Accuracy of Ensemble Classifier Before Hypertuning": "./images/accuracy_before_hypertuning.jpeg",
        "Distribution of Peptide Sequence Length": "./images/sequence_length_distribution.jpeg",
        "Distribution of Anti-Microbial and Non Anti-Microbial Peptides": "./images/antimicrobial_distribution.jpeg",
        "Accuracy After Hypertuning": "./images/accuracy_after_hypertuning.jpeg",
        "Confusion Matrix for Voting Classifier": "./images/confusion_matrix_voting_classifier.jpeg"
    }

    for title, img_path in image_files.items():
        if os.path.exists(img_path):
            st.subheader(title)
            st.image(Image.open(img_path))
        else:
            st.warning(f"Image '{img_path}' not found.")

# Home/Login and Prediction Page (protected by login)
elif page == "Login/Home":
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False

    # Login logic
    if not st.session_state.logged_in:
        st.title("Login")
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")

        if st.button("Login"):
            if authenticate(username, password):
                st.session_state.logged_in = True
                st.success("Login successful!")
            else:
                st.warning("Invalid credentials.")
    else:
        st.title("Ensemble Classifier for Antimicrobial Peptides")
        st.subheader("Enter Peptide Sequence for Prediction")
        peptide_seq = st.text_input("Peptide Sequence")

        if st.button("Predict"):
            if peptide_seq:
                result = predict_antimicrobial(peptide_seq)
                st.write(f"The prediction for the given sequence is: {'Antimicrobial' if result == 'Antimicrobial' else 'Non-Antimicrobial'}")
            else:
                st.warning("Please enter a peptide sequence.")

        if st.button("Logout"):
            st.session_state.logged_in = False
