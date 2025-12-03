# NLP Project Report: YouTube Spam Comment Classifier
## Katy Perry - Roar Video Comments

**Team Members:** [Add your names here]  
**Date:** December 3, 2025  
**Course:** COMP237 - Introduction to AI

---

## 1. Project Overview

### 1.1 Objective
The goal of this project is to build a text classifier using the **Bag of Words** language model and the **Naive Bayes** classifier to filter spam comments from YouTube videos. The classifier distinguishes between legitimate user comments and auto-generated spam comments.

### 1.2 Dataset
- **Source:** UCI Machine Learning Repository - YouTube Spam Collection
- **Movie:** Katy Perry - Roar
- **File:** Youtube02-KatyPerry.csv
- **Total Comments:** 350
- **Class Distribution:** 
  - Spam (CLASS=1): 175 comments (50%)
  - Non-Spam (CLASS=0): 175 comments (50%)
- **Features Used:** 
  - CONTENT: The text of the comments
  - CLASS: The spam label (0 = Non-Spam, 1 = Spam)

---

## 2. Methodology

### 2.1 Data Exploration
- Loaded data into pandas DataFrame
- Verified no missing values
- Analyzed class distribution (perfectly balanced dataset)
- Examined sample comments from both classes
- Generated class distribution visualization

**Key Observations:**
- Dataset is well-balanced (50/50 split)
- Spam comments typically contain URLs, promotional content, and calls to action
- Non-spam comments discuss video/song content naturally

### 2.2 Feature Engineering

#### Step 1: CountVectorizer (Bag of Words)
- Applied `count_vectorizer.fit_transform()` directly on raw text
- Converted text to numerical features using word frequencies
- **Results:**
  - Shape: (350, 1738)
  - 350 samples, 1738 unique words in vocabulary
  - Sparse matrix: 99.13% sparse (memory efficient)

#### Step 2: TF-IDF Transformation
- Applied Term Frequency-Inverse Document Frequency to downscale features
- Reduces importance of common words, increases weight of rare/unique words
- **Results:**
  - Shape: (350, 1738) - same dimensions, but weighted values
  - TF-IDF values range: 0.0 to 1.0 (mean: 0.0019)
  - Top weighted words in spam: "myleadergate", "pilot", "generate", "auto", "leads"

### 2.3 Data Preparation
- Shuffled dataset using `pandas.sample(frac=1, random_state=42)`
- Split data: 75% training (262 samples), 25% testing (88 samples)
- Training set: 135 spam, 127 non-spam
- Test set: 40 spam, 48 non-spam

### 2.4 Model Training
- **Algorithm:** Multinomial Naive Bayes
- Trained on 262 samples with TF-IDF features
- Training Accuracy: **99.24%**

---

## 3. Model Evaluation

### 3.1 Cross-Validation (5-Fold)
Performed 5-fold cross-validation on training data to assess model robustness:

| Metric | Value |
|--------|-------|
| Mean Accuracy | **90.45%** |
| Standard Deviation | 1.26% |
| Min Accuracy | 88.46% |
| Max Accuracy | 92.45% |

**Analysis:** Low standard deviation indicates the model is stable and generalizes well across different data splits.

### 3.2 Test Set Performance
- **Test Accuracy:** 90.91%

#### Confusion Matrix:
```
                Predicted
                Non-Spam  Spam
Actual Non-Spam    47      1
       Spam         7     33
```

**Breakdown:**
- True Negatives (Non-Spam correctly predicted): 47
- False Positives (Non-Spam predicted as Spam): 1
- False Negatives (Spam predicted as Non-Spam): 7
- True Positives (Spam correctly predicted): 33

#### Classification Metrics:

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Non-Spam | 0.87 | 0.98 | 0.92 | 48 |
| Spam | 0.97 | 0.82 | 0.89 | 40 |
| **Accuracy** | | | **0.91** | **88** |

**Key Insights:**
- **High Precision for Spam (97%):** When the model predicts spam, it's correct 97% of the time
- **Excellent Recall for Non-Spam (98%):** The model correctly identifies 98% of legitimate comments
- **Low False Positive Rate:** Only 1 legitimate comment incorrectly flagged as spam
- **Good Spam Detection:** Catches 82% of spam comments

---

## 4. Custom Comment Testing

### 4.1 Test Comments
We created 6 custom comments to test the model's generalization:

**Non-Spam Comments:**
1. "This was actually a great movie, very very instructive. I believe such content should be more promoted"
2. "Who else is here from december 2025? let's gather and give strenght to this channel."
3. "I promise to rewatch this movie over and over until I leave the earth"
4. "I really appreciated the content but I believe you should improve on the image quality."

**Spam Comments:**
5. "We are hiring urgently http://remitdol.hrfrim.ca"
6. "Divorced women are waiting for you in your area. Check how hot they are here hhttp://hottie.tv"

### 4.2 Results
- **Accuracy:** 66.67% (4 out of 6 correct)

**Correct Predictions:**
- ✓ Comment 1: Non-Spam → Predicted Non-Spam
- ✓ Comment 3: Non-Spam → Predicted Non-Spam
- ✓ Comment 4: Non-Spam → Predicted Non-Spam
- ✓ Comment 5: Spam → Predicted Spam

**Incorrect Predictions:**
- ✗ Comment 2: Non-Spam → Predicted Spam (False Positive)
- ✗ Comment 6: Spam → Predicted Non-Spam (False Negative)

### 4.3 Analysis
The model performed well on typical comments but struggled with:
- **Comment 2:** Flagged as spam possibly due to phrases like "gather" and "december 2025" which might resemble promotional patterns
- **Comment 6:** Dating/adult spam not well-represented in training data, missed due to unusual phrasing

---

## 5. Key Findings

### 5.1 Strengths
1. **High Overall Accuracy:** 90.91% on test data demonstrates strong performance
2. **Excellent Non-Spam Detection:** 98% recall minimizes false positives for legitimate users
3. **Efficient Feature Representation:** TF-IDF effectively captures spam patterns
4. **Robust Model:** Low variance in cross-validation (1.26% std) shows stability
5. **Strong Spam Precision:** 97% precision means high confidence when flagging spam

### 5.2 Spam Indicators Identified
- URLs and web links
- Promotional keywords: "free", "generate", "leads", "auto pilot"
- Call-to-action phrases: "subscribe", "check out", "please visit"
- Urgency words: "urgently", "waiting"

### 5.3 Limitations
1. **New Spam Patterns:** Model struggles with spam formats not in training data
2. **URL-Based False Positives:** Legitimate comments with URLs may be flagged
3. **Context Limitation:** Bag of Words doesn't capture word order or context
4. **Vocabulary Bound:** Limited to words seen during training
5. **Evolving Spam:** Spammers constantly change tactics to evade detection

---

## 6. Conclusions

The **Multinomial Naive Bayes classifier** combined with **Bag of Words** and **TF-IDF** successfully filters YouTube spam comments with **90.91% accuracy**. The model demonstrates:

- Strong performance on both training and test datasets
- Excellent recall (98%) for legitimate comments, minimizing user frustration
- High precision (97%) for spam detection, reducing false alarms
- Stable performance across different data splits (5-fold CV: 90.45% ± 1.26%)

The project successfully achieved its goal of building an effective spam filter using natural language processing techniques. The Bag of Words model, while simple, proves effective for this task by capturing the frequency patterns that distinguish spam from legitimate comments.

---

## 7. Recommendations for Improvement

1. **Expand Training Data:**
   - Include more diverse spam examples
   - Add comments from multiple video categories
   - Balance representation of different spam types

2. **Feature Engineering:**
   - Add URL count as a feature
   - Include comment length statistics
   - Consider capitalization patterns (spam often uses ALL CAPS)
   - Extract domain names from URLs as features

3. **Advanced Techniques:**
   - Implement ensemble methods (Random Forest, Gradient Boosting)
   - Try deep learning approaches (LSTM, BERT) for better context understanding
   - Use word embeddings (Word2Vec, GloVe) for semantic understanding
   - Consider bi-grams and tri-grams for phrase detection

4. **Model Maintenance:**
   - Regular retraining with new spam patterns
   - Active learning to incorporate misclassified examples
   - Monitor performance metrics over time
   - Implement feedback loop for continuous improvement

5. **Deployment Considerations:**
   - Set adjustable confidence thresholds
   - Implement human review for borderline cases
   - Create whitelist for trusted users
   - Log and analyze false positives/negatives

---

## 8. Technical Summary

| Aspect | Details |
|--------|---------|
| **Algorithm** | Multinomial Naive Bayes |
| **Feature Extraction** | CountVectorizer + TF-IDF |
| **Vocabulary Size** | 1,738 unique words |
| **Training Set** | 262 samples (75%) |
| **Test Set** | 88 samples (25%) |
| **Training Accuracy** | 99.24% |
| **Cross-Validation** | 90.45% (5-fold) |
| **Test Accuracy** | 90.91% |
| **Precision (Spam)** | 97% |
| **Recall (Non-Spam)** | 98% |

---

## 9. Visualizations Generated

1. **class_distribution.png** - Bar and pie charts showing balanced dataset
2. **confusion_matrix.png** - Heatmap of test set predictions

---

## 10. Files Delivered

1. `youtube_spam_classifier.py` - Complete implementation code
2. `PROJECT_REPORT.md` - This comprehensive report
3. `class_distribution.png` - Class distribution visualization
4. `confusion_matrix.png` - Confusion matrix visualization
5. `Youtube02-KatyPerry.csv` - Original dataset

---

**End of Report**
