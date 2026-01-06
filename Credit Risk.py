import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

train_path = 'C:/Users/Admin/Downloads/GiveMeSomeCredit/cs-training.csv'
test_path = 'C:/Users/Admin/Downloads/GiveMeSomeCredit/cs-test.csv'
train_df = pd.read_csv(train_path)
train_df = train_df.drop(columns=['Unnamed: 0'])

x_train = train_df.drop(columns=['SeriousDlqin2yrs'])
y_train = train_df['SeriousDlqin2yrs']

print('Training set shape:', train_df.shape)
print('Number of defaults:', y_train.sum())
print(train_df.head())


preprocessing = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())

])

#Logistic Regression
log_reg_model = Pipeline([
    ('preprocess', preprocessing),
    ('classifier',LogisticRegression(max_iter=1000, class_weight='balanced'))

])


cv= StratifiedKFold(n_splits=3, shuffle=True,random_state=42)
auc_scores= cross_val_score(log_reg_model, x_train, y_train, cv=cv, scoring='roc_auc')

print(f'Logistic Regression AUC: {auc_scores.mean():.3f} ± {auc_scores.std():.3f}')


#Random Forest

tree_preprocess = Pipeline([
     ('imputer', SimpleImputer(strategy='median'))
 ])

rf_model = Pipeline([
    ('preprocess', tree_preprocess),
    ('classifier', RandomForestClassifier(
        n_estimators=100, class_weight='balanced', random_state=42))
])
rf_scores = cross_val_score(rf_model, x_train, y_train, cv=cv, scoring='roc_auc')
print(f'Random Forest AUC: {rf_scores.mean():.3f} ± {rf_scores.std():.3f}')

#Gradient Boosting
gb_model = Pipeline([
    ('preprocess', tree_preprocess),
    ('classifier', GradientBoostingClassifier(
        n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42))
])

gb_scores = cross_val_score(gb_model, x_train, y_train, cv=cv, scoring='roc_auc')
print(f'Gradient Boosting AUC: {gb_scores.mean():.3f} ± {gb_scores.std():.3f}')




final_model = gb_model
final_model.fit(x_train, y_train)
print("Final model trained on the full dataset.")

import joblib
joblib.dump(final_model, 'CreditRisk_FinalModel.pkl')
print("Model saved as CreditRisk_FinalModel.pkl")

loaded_model = joblib.load('CreditRisk_FinalModel.pkl')
preds = loaded_model.predict_proba(x_train[:5])[:, 1]

readable_output = x_train[:5].copy()
readable_output['Predicted_Default_Risk'] = preds
readable_output['Risk_Label'] = (readable_output['Predicted_Default_Risk'] > 0.25).map({
    True: 'HIGH RISK',
    False: 'LOW RISK'
})

print(readable_output)





