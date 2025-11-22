# student_classification.py
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
import matplotlib.pyplot as plt
import pickle
import os

np.random.seed(42)

# --- 1) SYNTHESIZE dataset (1000 rows) roughly matching given distributions ---
n = 1000

# gender: female 52%, male 48%
genders = np.random.choice(['female','male'], size=n, p=[0.52,0.48])

# race/ethnicity - approximate
groups = np.random.choice(['group A','group B','group C','group D','group E'], size=n,
                          p=[0.07,0.12,0.32,0.26,0.23])

# parental education
parent_edu = np.random.choice(["some high school","high school","some college",
                               "associate's degree","bachelor's degree","master's degree"],
                               size=n, p=[0.05,0.13,0.23,0.22,0.20,0.17])

# lunch and test prep
lunch = np.random.choice(['standard','free/reduced'], size=n, p=[0.65,0.35])
test_prep = np.random.choice(['none','completed'], size=n, p=[0.64,0.36])

# helper to sample scores with a realistic multimodal distribution
def sample_scores(size):
    components = [
        (50, 8, 0.30),
        (65, 7, 0.40),
        (80, 6, 0.25),
        (92, 3, 0.05)
    ]
    comp = np.random.choice(len(components), size=size, p=[c[2] for c in components])
    result = np.zeros(size)
    for i, (mu, sigma, _) in enumerate(components):
        mask = comp==i
        result[mask] = np.random.normal(mu, sigma, size=mask.sum())
    return np.clip(result, 0, 100).round().astype(int)

math_scores = sample_scores(n)
reading_scores = (math_scores * 0.95 + np.random.normal(5,8,n)).clip(0,100).round().astype(int)
writing_scores = (math_scores * 0.9 + reading_scores * 0.1 + np.random.normal(3,7,n)).clip(0,100).round().astype(int)

# small boosts if test prep completed or higher parental education
for i in range(n):
    if test_prep[i]=='completed':
        math_scores[i] = min(100, math_scores[i] + np.random.randint(3,8))
        reading_scores[i] = min(100, reading_scores[i] + np.random.randint(2,7))
        writing_scores[i] = min(100, writing_scores[i] + np.random.randint(2,6))
    if "master" in parent_edu[i] or "bachelor" in parent_edu[i]:
        math_scores[i] = min(100, math_scores[i] + np.random.randint(0,5))

df = pd.DataFrame({
    'gender': genders,
    'race_ethnicity': groups,
    'parental_level_of_education': parent_edu,
    'lunch': lunch,
    'test_preparation_course': test_prep,
    'math score': math_scores,
    'reading score': reading_scores,
    'writing score': writing_scores
})

# Create average score and performance label
df['avg_score'] = df[['math score','reading score','writing score']].mean(axis=1).round(1)
def label_from_avg(x):
    if x < 50: return 'Low'
    if x < 70: return 'Average'
    return 'High'
df['performance'] = df['avg_score'].apply(label_from_avg)

# Save synthetic dataset for inspection
df.to_csv('synthetic_student_data.csv', index=False)
print("Saved synthetic_student_data.csv (first 6 rows):")
print(df.head(6))

# --- 2) Preprocessing and split ---
X = df.drop(columns=['avg_score','performance'])
y = df['performance']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, stratify=y, random_state=42)

cat_cols = ['gender','race_ethnicity','parental_level_of_education','lunch','test_preparation_course']
num_cols = ['math score','reading score','writing score']

preprocessor = ColumnTransformer(transformers=[
    ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols),

    ('num', StandardScaler(), num_cols)
], remainder='drop')

# --- 3) Pipelines & models ---
pipelines = {
    'logreg': Pipeline([('pre', preprocessor), ('clf', LogisticRegression(max_iter=1000))]),
    'rf': Pipeline([('pre', preprocessor), ('clf', RandomForestClassifier(random_state=42))]),
    'gb': Pipeline([('pre', preprocessor), ('clf', GradientBoostingClassifier(random_state=42))])
}

param_grids = {
    'logreg': {'clf__C':[0.1,1,10]},
    'rf': {'clf__n_estimators':[100,200], 'clf__max_depth':[6,12,None]},
    'gb': {'clf__n_estimators':[100,200], 'clf__learning_rate':[0.05,0.1], 'clf__max_depth':[3,5]}
}

best_models = {}
reports = {}

for name, pipe in pipelines.items():
    grid = GridSearchCV(pipe, param_grids[name], cv=4, scoring='f1_weighted', n_jobs=-1)
    grid.fit(X_train, y_train)
    best_models[name] = grid.best_estimator_
    preds = grid.predict(X_test)
    reports[name] = {
        'accuracy': accuracy_score(y_test, preds),
        'f1_weighted': f1_score(y_test, preds, average='weighted'),
        'classification_report': classification_report(y_test, preds, output_dict=True),
        'confusion_matrix': confusion_matrix(y_test, preds)
    }
    print(f"Model {name} done. Best params: {grid.best_params_}")

# --- 4) Summary & save best model ---
summary = pd.DataFrame([
    {'model': k, 'accuracy': reports[k]['accuracy'], 'f1_weighted': reports[k]['f1_weighted']}
    for k in reports
]).sort_values('f1_weighted', ascending=False)

print("\nModel comparison:\n", summary)

# Select best model
best_name = summary.iloc[0]['model']
best_model = best_models[best_name]

# ⭐ SAVE THE MODEL HERE
with open('best_student_model.pkl', 'wb') as f:
    pickle.dump(best_model, f)

print(f"\nBest model: {best_name}. Saved to best_student_model.pkl")


# Show classification report for best model
best_report = reports[best_name]['classification_report']
print("\nClassification report (best model):")
print(pd.DataFrame(best_report).transpose())

# --- 5) Plots ---
# Distribution of avg_score
plt.figure()
plt.hist(df['avg_score'], bins=20)
plt.title('Distribution of Average Scores')
plt.xlabel('Average score')
plt.ylabel('Count')
plt.grid(True)
plt.tight_layout()
plt.savefig('dist_avg_score.png')
plt.show()

# Boxplots
plt.figure()
plt.boxplot(df['math score'])
plt.title('Math score boxplot')
plt.ylabel('Score')
plt.tight_layout()
plt.savefig('box_math.png')
plt.show()

plt.figure()
plt.boxplot(df['reading score'])
plt.title('Reading score boxplot')
plt.ylabel('Score')
plt.tight_layout()
plt.savefig('box_reading.png')
plt.show()

plt.figure()
plt.boxplot(df['writing score'])
plt.title('Writing score boxplot')
plt.ylabel('Score')
plt.tight_layout()
plt.savefig('box_writing.png')
plt.show()

# Confusion matrix (best model)
cm = reports[best_name]['confusion_matrix']
plt.figure()
plt.imshow(cm, interpolation='nearest')
plt.title(f'Confusion Matrix - {best_name}')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.colorbar()
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, cm[i,j], ha="center", va="center")
plt.tight_layout()
plt.savefig('confusion_matrix.png')
plt.show()

# Feature importance (if tree-based)
if best_name in ['rf','gb']:
    clf = best_model.named_steps['clf']
    ohe = best_model.named_steps['pre'].named_transformers_['cat']
    cat_names = ohe.get_feature_names_out(cat_cols)
    feat_names = list(cat_names) + num_cols
    importances = clf.feature_importances_
    fi_df = pd.DataFrame({'feature':feat_names, 'importance':importances}).sort_values('importance', ascending=False).head(20)
    print("\nTop feature importances:")
    print(fi_df)
    plt.figure()
    plt.bar(fi_df['feature'], fi_df['importance'])
    plt.title('Top Feature Importances')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    plt.show()

print("\nSaved files: synthetic_student_data.csv, best_student_model.pkl, and the generated PNG plots.")
