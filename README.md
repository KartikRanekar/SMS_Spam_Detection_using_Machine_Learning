# SMS Spam Detection with Machine Learning

A foundational Natural Language Processing (NLP) classification project designed to identify SMS text messages as either 'Spam' (unsolicited junk/malicious) or 'Ham' (legitimate user messages). 

## 🎯 Objective
To construct a text-based binary classifier that extracts features from raw SMS strings and utilizes probabilistic or gradient-based machine learning models to effectively filter out spam messages.

## 🛠️ Technologies & Libraries
* **Language:** Python 3.x
* **Data Handling:** Pandas, NumPy
* **NLP & Text Vectorization:** Scikit-Learn (`CountVectorizer`, `TfidfTransformer`)
* **Machine Learning:** Scikit-Learn (Naive Bayes, SGD Classifier)
* **Visualization:** Matplotlib, Seaborn

## 🧠 Methodology
1. **Exploratory Data Analysis (EDA):**
   * Inspecting dataset class distributions (Spam vs. Ham proportions).
   * Feature engineering: Extracting message length characters to identify structural differences between spam and legitimate messages.
   * Visualizing message length distributions using histograms.
2. **Text Preprocessing:**
   * Removing punctuation and standardizing text.
   * Stripping common English stopwords.
3. **Vectorization pipeline (Bag of Words):**
   * Utilizing `CountVectorizer` to build a vocabulary dictionary and convert messages into token counts.
   * Translating counts into normalized TF-IDF (Term Frequency-Inverse Document Frequency) scores to downweight overly frequent words.
4. **Model Training:**
   * **Multinomial Naive Bayes:** A highly effective, probabilistic baseline model for text classification.
   * **Stochastic Gradient Descent (SGD) Classifier:** Evaluated as an alternative linear classifier.
5. **Evaluation:**
   * Assessing the model against testing datasets using Accuracy, Precision, Recall, and F1-Score to ensure minimal false positives (accidentally filtering legitimate messages).

## 🚀 How to Run
1. Install standard ML dependencies: `pip install pandas numpy scikit-learn matplotlib seaborn`
2. Open `SMS Spam Detection with Machine Learning.ipynb`.
3. Run the notebook cells to view the EDA visualizations, understand the Bag of Words matrix creation, and evaluate the classifier's performance.
