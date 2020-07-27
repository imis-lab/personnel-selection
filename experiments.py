# %%
import pandas as pd
from sklearn.model_selection import train_test_split

import classification
import evaluation
import utils

# %%
issues_df = pd.read_csv('dataset.csv')
included_developers = utils.get_top_n_most_frequent_labels(issues_df, 'label', 21)
issues_df = issues_df[issues_df['label'].isin(included_developers)]
print(len(issues_df))
# %%
le = utils.fit_label_encoder(issues_df, 'label')
issues_df['encoded_label'] = le.transform(issues_df['label'])

# %%
preprocessor = utils.TextPreprocessor()
issues_df['text'] = issues_df['text'].apply(preprocessor.clean_text)
# %%
## Example

# %%
train_issues_df, test_issues_df = train_test_split(issues_df, test_size=0.33, random_state=42)
fe_generator = utils.FeatureExtractorGenerator()
cv = fe_generator.bag_of_words_vectorizer(train_issues_df, 'text', stop_words='english', max_df=0.6)
x_train_transformed = cv.transform(train_issues_df['text'])
x_test_transformed = cv.transform(test_issues_df['test'])
# %%
clf_generator = classification.ClassifierGenerator()
classifiers = [
    ('Logistic Regression', clf_generator.logistic_regression()),
    ('Linear SVC', clf_generator.linear_svm()),
    ('Naive Bayes', clf_generator.naive_bayes()),
]

evaluator = evaluation.ModelEvaluator()
for name, clf in classifiers:
    print(name)
    print(evaluator.evaluate_classifier(clf, x_train_transformed, train_issues_df['encoded_label'], x_test_transformed,
                                        test_issues_df['encoded_label'])['accuracy'])
