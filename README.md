# spamEmailClassifier

Spam Email Classifier

This project is a machine learning-based spam email classifier implemented in Python. It utilizes Natural Language Processing (NLP) techniques to categorize emails as either spam or ham (non-spam) using a Logistic Regression model.

Features
* Data Preprocessing: Loads and processes email data from a CSV file.
* Text Vectorization: Transforms email text into numerical feature representations using Term Frequency-Inverse Document Frequency (TF-IDF).
* Model Training: Implements a Logistic Regression model for classification.
* Performance Evaluation: Assesses model accuracy on both training and test datasets.
* User Input Classification: Enables real-time classification of user-provided email samples.

Technologies Used
* Python
* Pandas (for data manipulation)
* NumPy (for numerical computations)
* Matplotlib & Seaborn (for data visualization)
* Scikit-learn (for machine learning and NLP processing)

Installation & Setup
1. Clone the repository: git clone https://github.com/your-repo/spam-classifier.git
2. cd spam-classifier
3. Install the required dependencies: pip install numpy pandas matplotlib seaborn scikit-learn
4. Ensure that mail_data.csv is present in the project directory.
5. Execute the script: python spam_classifier.py

How It Works
1. Data Loading: Reads mail_data.csv, which contains emails labeled as spam or ham.
2. Data Preprocessing:
    * Converts the categorical 'Category' column into numerical values (spam → 0, ham → 1).
    * Extracts email messages for training and testing.
3. Text Vectorization:
    * Uses TF-IDF to convert email text into a numerical feature matrix.
4. Model Training:
    * A Logistic Regression classifier is trained on the processed data.
5. Model Evaluation:
    * Measures the model's accuracy on both training and test datasets.
6. User Input Classification:
    * Accepts user-provided email text and classifies it as spam or ham.

Example Usage
mails = ["From, 6 days, 16+ Ts and Cs apply. Reply HL for info"]
mails_features = feature_extraction.transform(mails)
res = logRegModel.predict(mails_features)
* If 0, the email is classified as Spam.
* If 1, the email is classified as Ham.

Results
* The model achieves a measurable accuracy score on both training and test datasets, which is displayed upon execution.


