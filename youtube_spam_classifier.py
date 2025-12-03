"""
NLP Project: YouTube Spam Comment Classifier
Movie: Katy Perry - Roar
Using Naive Bayes Classifier with Bag of Words Model

Team Members: [Add your names here]
Date: December 3, 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import warnings
warnings.filterwarnings('ignore')

# ANSI color codes for terminal output
GREEN = '\033[92m'
RESET = '\033[0m'

# Download required NLTK data
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

print(f"\n{GREEN}STEP 1: Load Data and Basic Exploration{RESET}\n")

# 1. Load the data into a pandas dataframe
df = pd.read_csv('Youtube02-KatyPerry.csv')

# 2. Basic data exploration
print(f"{GREEN}Dataset Overview:{RESET}")
print(f"Shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")

print(f"\n{GREEN}First 5 rows:{RESET}")
print(df.head())

print(f"\n{GREEN}Data Info:{RESET}")
print(df.info())

print(f"\n{GREEN}Missing Values:{RESET}")
print(df.isnull().sum())

print(f"\n{GREEN}Class Distribution:{RESET}")
print(df['CLASS'].value_counts())
print(f"\n{df['CLASS'].value_counts(normalize=True) * 100}")

# Visualize class distribution
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
df['CLASS'].value_counts().plot(kind='bar', color=['green', 'red'])
plt.title('Class Distribution')
plt.xlabel('Class (0=Non-Spam, 1=Spam)')
plt.ylabel('Count')
plt.xticks(rotation=0)

plt.subplot(1, 2, 2)
df['CLASS'].value_counts().plot(kind='pie', autopct='%1.1f%%', colors=['green', 'red'])
plt.title('Class Distribution')
plt.ylabel('')
plt.legend(['Non-Spam (0)', 'Spam (1)'])
plt.tight_layout()
plt.savefig('class_distribution.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"\n{GREEN}Two columns identified for the project:{RESET}")
print("1. CONTENT - The text of the comments (features)")
print("2. CLASS - The spam label (0=Non-Spam, 1=Spam)")

print(f"\n{GREEN}Sample Comments:{RESET}")
print("\nNon-Spam (CLASS=0):")
for i, comment in enumerate(df[df['CLASS']==0]['CONTENT'].head(2).values, 1):
    print(f"{i}. {comment[:100]}...")
print("\nSpam (CLASS=1):")
for i, comment in enumerate(df[df['CLASS']==1]['CONTENT'].head(2).values, 1):
    print(f"{i}. {comment[:100]}...")

print(f"\n{GREEN}Step 1 Complete!{RESET}")

# ============================================================================
# STEP 2: Apply CountVectorizer (Bag of Words)
# ============================================================================

print(f"\n{GREEN}STEP 2: Apply CountVectorizer - Bag of Words Model{RESET}\n")

# Extract the two columns needed
X = df['CONTENT']  # Features (text)
y = df['CLASS']    # Target (labels)

# Initialize CountVectorizer
count_vectorizer = CountVectorizer()

# Fit and transform the text data
print(f"{GREEN}Applying count_vectorizer.fit_transform()...{RESET}")
X_counts = count_vectorizer.fit_transform(X)

# Present highlights of the output (initial features)
print(f"\n{GREEN}CountVectorizer Results (Initial Features):{RESET}")
print(f"Shape of transformed data: {X_counts.shape}")
print(f"Number of samples: {X_counts.shape[0]}")
print(f"Number of features (unique words): {X_counts.shape[1]}")
print(f"Data type: {type(X_counts)}")
print(f"Matrix sparsity: {(1 - X_counts.nnz / (X_counts.shape[0] * X_counts.shape[1])) * 100:.2f}%")

# Show some feature names (words in vocabulary)
print(f"\n{GREEN}Sample of vocabulary (first 20 words):{RESET}")
feature_names = count_vectorizer.get_feature_names_out()
print(feature_names[:20])

print(f"\n{GREEN}Sample of vocabulary (last 20 words):{RESET}")
print(feature_names[-20:])

print(f"\n{GREEN}Total vocabulary size:{RESET} {len(feature_names)} unique words")

print(f"\n{GREEN}Step 2 Complete!{RESET}")

# ============================================================================
# STEP 3: Apply TF-IDF Transformation
# ============================================================================

print(f"\n{GREEN}STEP 3: Apply TF-IDF Transformation (Downscaling){RESET}\n")

# Initialize TF-IDF Transformer
tfidf_transformer = TfidfTransformer()

# Transform the count data to TF-IDF
print(f"{GREEN}Applying TF-IDF transformation...{RESET}")
X_tfidf = tfidf_transformer.fit_transform(X_counts)

# Present highlights of the output (final features)
print(f"\n{GREEN}TF-IDF Results (Final Features):{RESET}")
print(f"Shape of TF-IDF data: {X_tfidf.shape}")
print(f"Number of samples: {X_tfidf.shape[0]}")
print(f"Number of features: {X_tfidf.shape[1]}")
print(f"Data type: {type(X_tfidf)}")
print(f"Matrix sparsity: {(1 - X_tfidf.nnz / (X_tfidf.shape[0] * X_tfidf.shape[1])) * 100:.2f}%")

# Show some statistics about TF-IDF values
print(f"\n{GREEN}TF-IDF Value Statistics:{RESET}")
print(f"Min TF-IDF value: {X_tfidf.min():.6f}")
print(f"Max TF-IDF value: {X_tfidf.max():.6f}")
print(f"Mean TF-IDF value: {X_tfidf.mean():.6f}")

# Show example of TF-IDF values for first comment
print(f"\n{GREEN}Example - First comment TF-IDF (top 10 weighted words):{RESET}")
first_comment_tfidf = X_tfidf[0].toarray()[0]
top_indices = first_comment_tfidf.argsort()[-10:][::-1]
for idx in top_indices:
    if first_comment_tfidf[idx] > 0:
        print(f"  {feature_names[idx]}: {first_comment_tfidf[idx]:.4f}")

print(f"\n{GREEN}Step 3 Complete!{RESET}")

# ============================================================================
# STEP 4: Shuffle and Split Dataset (75% Train, 25% Test)
# ============================================================================

print(f"\n{GREEN}STEP 4: Shuffle and Split Dataset{RESET}\n")

# Shuffle the dataset using pandas.sample with frac=1
print(f"{GREEN}Shuffling dataset with pandas.sample(frac=1)...{RESET}")
df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Get the shuffled indices to apply to the TF-IDF matrix
shuffled_indices = df.sample(frac=1, random_state=42).index

# Apply shuffling to X_tfidf and y
X_tfidf_shuffled = X_tfidf[shuffled_indices]
y_shuffled = y.iloc[shuffled_indices].reset_index(drop=True)

# Split the data: 75% training, 25% testing
split_point = int(0.75 * len(df_shuffled))

print(f"\n{GREEN}Splitting data (75% train, 25% test):{RESET}")
print(f"Total samples: {len(df_shuffled)}")
print(f"Training samples: {split_point}")
print(f"Testing samples: {len(df_shuffled) - split_point}")

# Split features (X) and labels (y)
X_train = X_tfidf_shuffled[:split_point]
X_test = X_tfidf_shuffled[split_point:]
y_train = y_shuffled[:split_point]
y_test = y_shuffled[split_point:]

print(f"\n{GREEN}Training set:{RESET}")
print(f"X_train shape: {X_train.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"Class distribution in training: {y_train.value_counts().to_dict()}")

print(f"\n{GREEN}Testing set:{RESET}")
print(f"X_test shape: {X_test.shape}")
print(f"y_test shape: {y_test.shape}")
print(f"Class distribution in testing: {y_test.value_counts().to_dict()}")

print(f"\n{GREEN}Step 4 Complete!{RESET}")

# ============================================================================
# STEP 5: Train Naive Bayes Classifier
# ============================================================================

print(f"\n{GREEN}STEP 5: Train Naive Bayes Classifier{RESET}\n")

# Initialize the Naive Bayes classifier (MultinomialNB for text)
classifier = MultinomialNB()

# Train the classifier on the training data
print(f"{GREEN}Training Naive Bayes classifier...{RESET}")
classifier.fit(X_train, y_train)

# Make predictions on training data to check training accuracy
y_train_pred = classifier.predict(X_train)
train_accuracy = accuracy_score(y_train, y_train_pred)

print(f"\n{GREEN}Training Results:{RESET}")
print(f"Training Accuracy: {train_accuracy * 100:.2f}%")
print(f"Classifier type: {type(classifier).__name__}")
print(f"Number of classes: {len(classifier.classes_)}")
print(f"Classes: {classifier.classes_}")

print(f"\n{GREEN}Step 5 Complete!{RESET}")

# ============================================================================
# STEP 6: Cross-Validate Model (5-Fold)
# ============================================================================

print(f"\n{GREEN}STEP 6: Cross-Validate Model on Training Data{RESET}\n")

# Perform 5-fold cross-validation on training data
print(f"{GREEN}Performing 5-fold cross-validation...{RESET}")
cv_scores = cross_val_score(classifier, X_train, y_train, cv=5, scoring='accuracy')

print(f"\n{GREEN}Cross-Validation Results:{RESET}")
print(f"Accuracy scores for each fold: {cv_scores}")
print(f"Mean Accuracy: {cv_scores.mean() * 100:.2f}%")
print(f"Standard Deviation: {cv_scores.std() * 100:.2f}%")
print(f"Min Accuracy: {cv_scores.min() * 100:.2f}%")
print(f"Max Accuracy: {cv_scores.max() * 100:.2f}%")

print(f"\n{GREEN}Step 6 Complete!{RESET}")

# ============================================================================
# STEP 7: Test Model and Evaluate Performance
# ============================================================================

print(f"\n{GREEN}STEP 7: Test Model on Test Data{RESET}\n")

# Make predictions on test data
print(f"{GREEN}Making predictions on test data...{RESET}")
y_test_pred = classifier.predict(X_test)

# Calculate accuracy
test_accuracy = accuracy_score(y_test, y_test_pred)

print(f"\n{GREEN}Test Results:{RESET}")
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

# Confusion Matrix
print(f"\n{GREEN}Confusion Matrix:{RESET}")
cm = confusion_matrix(y_test, y_test_pred)
print(cm)
print(f"\nConfusion Matrix Breakdown:")
print(f"True Negatives (Non-Spam correctly predicted): {cm[0][0]}")
print(f"False Positives (Non-Spam predicted as Spam): {cm[0][1]}")
print(f"False Negatives (Spam predicted as Non-Spam): {cm[1][0]}")
print(f"True Positives (Spam correctly predicted): {cm[1][1]}")

# Classification Report
print(f"\n{GREEN}Classification Report:{RESET}")
print(classification_report(y_test, y_test_pred, target_names=['Non-Spam', 'Spam']))

# Visualize Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Non-Spam', 'Spam'], yticklabels=['Non-Spam', 'Spam'])
plt.title('Confusion Matrix - Test Data')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"\n{GREEN}Step 7 Complete!{RESET}")

# ============================================================================
# STEP 8: Test with Custom Comments
# ============================================================================

print(f"\n{GREEN}STEP 8: Test Classifier with 6 Custom Comments{RESET}\n")

# Define 6 new custom comments (4 non-spam, 2 spam)
custom_comments = [
    # Non-Spam Comments
    "This was actually a great movie, very very instructive. I believe such content should be more promoted",
    "Who else is here from december 2025? let's gather and give strenght to this channel.",
    "I promise to rewatch this movie over and over until I leave the earth",
    "I really appreciated the content but I believe you should improve on the image quality.",
    # Spam Comments
    "We are hiring urgently http://remitdol.hrfrim.ca",
    "Divorced women are waiting for you in your area. Check how hot they are here hhttp://hottie.tv"
]

# Expected labels (for verification)
expected_labels = [0, 0, 0, 0, 1, 1]  # 0 = Non-Spam, 1 = Spam
label_names = {0: 'Non-Spam', 1: 'Spam'}

# Transform custom comments using the same vectorizer and transformer
print(f"{GREEN}Transforming custom comments...{RESET}")
custom_counts = count_vectorizer.transform(custom_comments)
custom_tfidf = tfidf_transformer.transform(custom_counts)

# Make predictions
custom_predictions = classifier.predict(custom_tfidf)

# Display results
print(f"\n{GREEN}Prediction Results:{RESET}\n")
for i, (comment, expected, predicted) in enumerate(zip(custom_comments, expected_labels, custom_predictions), 1):
    status = "✓ CORRECT" if expected == predicted else "✗ INCORRECT"
    print(f"Comment {i}:")
    print(f"  Text: {comment}")
    print(f"  Expected: {label_names[expected]}")
    print(f"  Predicted: {label_names[predicted]}")
    print(f"  Status: {status}")
    print()

# Calculate accuracy on custom comments
custom_accuracy = accuracy_score(expected_labels, custom_predictions)
print(f"{GREEN}Accuracy on Custom Comments: {custom_accuracy * 100:.2f}%{RESET}")
print(f"Correct predictions: {sum(expected_labels == custom_predictions)}/6")

print(f"\n{GREEN}Step 8 Complete!{RESET}")
