## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn

## Dataset

The dataset consists of text data labeled with video domain categories (e.g., AI , Cyber security ,.. ). It is preprocessed and split into features (text) and labels (video categories) for training the classification models.

## Approach

1. **Data Preprocessing**: Clean and preprocess the text data, including tokenization, removing stopwords, and vectorizing text using TF-IDF (Term Frequency-Inverse Document Frequency).

2. **Feature Selection**: Select relevant features using techniques such as chi-square test to identify the most informative features for classification.

3. **Model Selection**: Evaluate the performance of different classification algorithms, including Logistic Regression, Multinomial Naive Bayes, Random Forest Classifier, and Linear Support Vector Classifier (LinearSVC).

4. **Model Training and Evaluation**: Train each model on the training data and evaluate its performance using accuracy metrics on both training and testing datasets.

5. **Selection of Final Model**: Choose the model with the highest accuracy on the testing dataset as the final model for video segmentation.

## Model Evaluation

The following classification algorithms were tested for sentiment analysis:

- Logistic Regression
- Multinomial Naive Bayes
- Random Forest Classifier
- Linear Support Vector Classifier (LinearSVC)

After evaluating the performance of each model, LinearSVC was selected as the final model due to its high accuracy on the testing dataset.

## Usage

1. **Data Splitting**: Split the dataset into training and testing sets.

2. **Model Training**: Train each classification model using the training dataset.

3. **Model Evaluation**: Evaluate the performance of each model using accuracy metrics on both training and testing datasets.

4. **Final Model Selection**: Choose the model with the highest accuracy on the testing dataset as the final model for sentiment analysis.
