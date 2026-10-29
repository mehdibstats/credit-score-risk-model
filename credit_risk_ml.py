#%%
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score, cross_val_predict
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import RandomizedSearchCV
df = pd.read_csv("train.csv", decimal=".")
y = df['Credit_Score']
#%%
LOAN_TYPES = ['Auto Loan', 'Credit-Builder Loan', 'Debt Consolidation Loan',
              'Home Equity Loan', 'Mortgage Loan', 'Payday Loan',
              'Personal Loan', 'Student Loan']

def clean(df):
    df = df.copy()
    new_col = []
    new_prev_col = []
    #Colonne a supprimer
    drop_col = ['Name', 'SSN','ID','Credit_Score']
    #colonne numerique 
    num_col = ['Age', 'Annual_Income', 'Monthly_Inhand_Salary', 'Num_Bank_Accounts',
           'Num_Credit_Card', 'Interest_Rate', 'Num_of_Loan',
           'Delay_from_due_date', 'Num_of_Delayed_Payment', 'Changed_Credit_Limit',
           'Num_Credit_Inquiries', 'Outstanding_Debt','Credit_Utilization_Ratio',
           'Total_EMI_per_month','Amount_invested_monthly', 'Monthly_Balance']
    #Colonne categorielle textuelle 
    group_col = ['Customer_ID']
    exlu_col = ['Type_of_Loan','Credit_History_Age']
    cat_col = df.columns.difference(num_col + drop_col + group_col+ exlu_col).to_list()
    #transformation en to_numeric de la colonne num_col 
    for col in num_col:
        df[col] = pd.to_numeric(df[col].astype(str).str.replace('_','', regex=False), errors= 'coerce')
    #Filtrage des valeurs aberrantes par colonne
    for col in ['Num_Bank_Accounts', 'Num_Credit_Card', 'Num_of_Loan', 'Num_of_Delayed_Payment']:
        df.loc[df[col] < 0, col] = np.nan
    df.loc[df['Amount_invested_monthly'] == 10000, 'Amount_invested_monthly'] = np.nan
    df.loc[df['Monthly_Balance'] < -1e6, 'Monthly_Balance'] = np.nan
    df.loc[(df['Age'] > 110) | (df['Age'] < 14), 'Age'] = np.nan
    rf_AI = df.groupby('Customer_ID')['Annual_Income'].transform('median')
    mask_AI = df['Annual_Income'] > 5 * rf_AI
    df.loc[mask_AI, 'Annual_Income'] = np.nan 
    df.loc[df['Num_Bank_Accounts'] > 15, 'Num_Bank_Accounts'] = np.nan
    df.loc[df['Num_Credit_Card'] > 15, 'Num_Credit_Card'] = np.nan
    df.loc[df['Interest_Rate'] > 34, 'Interest_Rate'] = np.nan
    df.loc[df['Num_of_Loan'] > 9, 'Num_of_Loan'] = np.nan
    df.loc[df['Num_of_Delayed_Payment'] > 30, 'Num_of_Delayed_Payment'] = np.nan
    df.loc[df['Num_Credit_Inquiries'] > 15, 'Num_Credit_Inquiries'] = np.nan
    ref = df.groupby('Customer_ID')['Total_EMI_per_month'].transform('median')
    mask = df['Total_EMI_per_month'] > 5 * ref
    df.loc[mask, 'Total_EMI_per_month'] = np.nan    
    #imputation par la median 
    med_col = ['Age', 'Annual_Income', 'Monthly_Inhand_Salary','Interest_Rate',
           'Delay_from_due_date', 'Changed_Credit_Limit',
            'Outstanding_Debt','Credit_Utilization_Ratio',
            'Total_EMI_per_month','Amount_invested_monthly', 'Monthly_Balance']
    for col in med_col:
        df[col] = df[col].fillna(df.groupby('Customer_ID')[col].transform('median'))
    #imputation a la plus frequente 
    freq_col = [c for c in num_col if c not in med_col]
    for col in freq_col:
        df[col] = df[col].fillna(
            df.groupby('Customer_ID')[col].transform(
            lambda s: s.mode().iloc[0] if not s.mode().empty else np.nan
        )
    )
    agg_col = ['Delay_from_due_date', 'Num_of_Delayed_Payment',
    'Num_Credit_Inquiries', 'Credit_Utilization_Ratio',
    'Monthly_Balance']
    stats = ['mean']

    for col in agg_col: 
        for stat in stats:
            df[f'{col}_{stat}'] = df.groupby('Customer_ID')[col].transform(stat)
            new_col.append(f'{col}_{stat}')
        df[f'{col}_dev'] = df[col] - df[f'{col}_{stat}']
        new_col.append(f'{col}_dev')
    #decalage des colonnes par rapport au mois 
    prev_col = ['Delay_from_due_date','Credit_Utilization_Ratio','Num_of_Delayed_Payment']
    for col in prev_col: 
        df[f'{col}_prev'] = df.groupby('Customer_ID')[col].shift(1)
        new_prev_col.append(f'{col}_prev')
    #labellisé les differents type de credit 
    df['Type_of_Loan'] = df['Type_of_Loan'].replace('Not Specified', np.nan)
    mode_loan = df.groupby('Customer_ID')['Type_of_Loan'].agg(
        lambda s: s.mode().iloc[0] if not s.mode().empty else np.nan)
    df['Type_of_Loan'] = df['Type_of_Loan'].fillna(df['Customer_ID'].map(mode_loan))
    s = (df['Type_of_Loan'].fillna('').str.replace(' and ', ', '))
    loan_lists = s.str.split(',')
    loan_lists = loan_lists.apply(lambda lst: [l.strip() for l in lst if l.strip()])
    mlb = MultiLabelBinarizer(classes=LOAN_TYPES)
    loans = pd.DataFrame(mlb.fit_transform(loan_lists),
                         columns= mlb.classes_,
                        index= df.index).add_prefix('Loan_')
    df = pd.concat([df, loans], axis = 1)
    cat_col.remove('Month')
    df['Credit_Mix'] = df['Credit_Mix'].replace('_', np.nan)
    df['Occupation'] = df['Occupation'].replace('_______', np.nan)
    df['Payment_Behaviour'] = df['Payment_Behaviour'].replace('!@9#%8', np.nan)
    df['Payment_of_Min_Amount'] = df['Payment_of_Min_Amount'].replace('NM', np.nan)
    #imputation au plus frequent
    for col in cat_col:
        df[col] = df[col].fillna(
            df.groupby('Customer_ID')[col].transform(
            lambda s: s.mode().iloc[0] if not s.mode().empty else np.nan
        )
    )
    h = df['Credit_History_Age'].str.extract(r'(\d+) Years and (\d+) Months').astype(float)
    df['Credit_History_Age'] = h[0] * 12 + h[1]
    df['Credit_History_Age'] = df['Credit_History_Age'].fillna(
        df.groupby('Customer_ID')['Credit_History_Age'].transform('median'))
    num_col.append('Credit_History_Age')
    loan_col = list(loans.columns)
    num_col += new_col + new_prev_col
    features = num_col + cat_col + loan_col
    return df[features], num_col, cat_col, loan_col

#%%
#verification de la repartition via k-fold 
X, num_col, cat_col, loan_col = clean(df)
groups = df['Customer_ID']
sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

for fold, (train_idx, test_idx) in enumerate(sgkf.split(X, y, groups=groups)):
    print(f'--- Pli {fold + 1} ---')
    print('Taille train / test :', len(train_idx), len(test_idx))
    print('Clients test :', groups.iloc[test_idx].nunique())
    print('Répartition classes test :')
    print(y.iloc[test_idx].value_counts(normalize=True).round(2))
    # contrôle anti-fuite : aucun client en commun
    print('Clients communs train/test :',
          len(set(groups.iloc[train_idx]) & set(groups.iloc[test_idx])))

#%%
num_pip = Pipeline([
    ('imp', SimpleImputer(strategy='median')),
    ('sc', StandardScaler()),
])
cat_pip = Pipeline([
    ('imp', SimpleImputer(strategy='most_frequent')),
    ('oh', OneHotEncoder(handle_unknown='ignore'))
])
preprocess = ColumnTransformer([
    ('num', num_pip, num_col),
    ('cat', cat_pip, cat_col),
    ('loan', 'passthrough', loan_col)
])
model = Pipeline([
    ('prep', preprocess),
    ('model', HistGradientBoostingClassifier(
    learning_rate=0.05, max_iter=300,
    max_leaf_nodes=31, min_samples_leaf=50,
    early_stopping=False, class_weight='balanced', random_state=42))
])
scores = cross_val_score(model, X, y, cv=sgkf, groups=groups, scoring='f1_macro')
scores.mean()
# %%
y_pred = cross_val_predict(model, X, y, cv=sgkf, groups=groups)
labels = sorted(y.unique())
cm = confusion_matrix(y, y_pred, labels=labels, normalize= 'true')

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
disp.plot(cmap=plt.cm.Blues)
plt.show()
# %%
y_proba = cross_val_predict(model, X, y, cv=sgkf, groups=groups, method='predict_proba')
auc = roc_auc_score(y, y_proba, multi_class='ovr', average='macro')
auc
# %%

param_dist = {
    'model__learning_rate': [0.03, 0.05, 0.08, 0.1],
    'model__max_iter': [200, 300, 500],
    'model__max_leaf_nodes': [15, 31, 63],
    'model__min_samples_leaf': [20, 50, 100],
    'model__l2_regularization': [0, 0.1, 1.0],
    'model__class_weight': [None, 'balanced'],
}

search = RandomizedSearchCV(
    model, param_dist, n_iter=10,
    cv=sgkf, scoring='f1_macro',
    n_jobs=-1, random_state=42, verbose=1)

search.fit(X, y, groups=groups)
print(search.best_score_, search.best_params_)
# %%
from sklearn.metrics import classification_report, roc_auc_score

# 1. Évaluation de la config finale (hors échantillon, par clients)
best = search.best_estimator_

y_pred = cross_val_predict(best, X, y, cv=sgkf, groups=groups)
print(classification_report(y, y_pred))

y_proba = cross_val_predict(best, X, y, cv=sgkf, groups=groups, method='predict_proba')
print('AUC :', roc_auc_score(y, y_proba, multi_class='ovr', average='macro'))

labels = sorted(y.unique())
cm = confusion_matrix(y, y_pred, labels=labels, normalize='true')
ConfusionMatrixDisplay(cm, display_labels=labels).plot(cmap=plt.cm.Blues)
plt.show()

# 2. Préparation du test
test = pd.read_csv('test.csv')
X_test, *_ = clean(test)
print('Colonnes identiques :', list(X_test.columns) == list(X.columns))

# 3. Prédiction et export
pred = best.predict(X_test)
submission = pd.DataFrame({'ID': test['ID'], 'Credit_Score': pred})
submission.to_csv('submission.csv', index=False)

# 4. Contrôle des proportions
print(submission['Credit_Score'].value_counts(normalize=True).round(2))
print(y.value_counts(normalize=True).round(2))

# %%
from sklearn.inspection import permutation_importance
import joblib
#Check de submission.csv
sub = pd.read_csv('submission.csv')
print(sub.shape)                      # attendu : (50000, 2)
print(sub.isna().sum())               # attendu : 0 partout
print(sub['ID'].is_unique)            # attendu : True
print(sub['Credit_Score'].value_counts())
#sauvegarder le modele
joblib.dump(best, 'model.joblib')

# %%
from sklearn.base import clone
from sklearn.inspection import permutation_importance

# 1. Un découpage par clients : on prend le premier pli
train_idx, val_idx = next(sgkf.split(X, y, groups=groups))

# 2. On entraîne une copie du modèle sur la partie train seulement
m = clone(best)
m.fit(X.iloc[train_idx], y.iloc[train_idx])

# 3. Importance calculée sur les clients que le modèle n'a pas vus
perm = permutation_importance(
    m, X.iloc[val_idx], y.iloc[val_idx],
    scoring='f1_macro', n_repeats=5, random_state=42, n_jobs=2)

# 4. Résultat trié et graphique des 15 premières
imp = pd.Series(perm.importances_mean, index=X.columns).sort_values(ascending=False)
print(imp.head(15))
imp.head(15).plot.barh(figsize=(8, 6)).invert_yaxis()
plt.xlabel('Baisse du macro-F1 quand la variable est mélangée')
plt.show()
# %%
