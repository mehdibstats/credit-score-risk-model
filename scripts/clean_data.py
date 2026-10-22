"""Data cleaning for the Credit Score dataset.

The cleaning logic is the one written in the project notebook, wrapped in one
function so it can be called from run_all.py. Every imputation uses only the rows
of the same customer (groupby Customer_ID), so no information moves between
customers. The remaining missing values are handled later, in the sklearn
pipeline, fitted on the training set only.
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer


def clean(df):
    # Columns to drop / numeric / categorical
    drop_col = ['Name', 'SSN', 'ID', 'Credit_Score']
    num_col = ['Age', 'Annual_Income', 'Monthly_Inhand_Salary', 'Num_Bank_Accounts',
               'Num_Credit_Card', 'Interest_Rate', 'Num_of_Loan',
               'Delay_from_due_date', 'Num_of_Delayed_Payment', 'Changed_Credit_Limit',
               'Num_Credit_Inquiries', 'Outstanding_Debt', 'Credit_Utilization_Ratio',
               'Total_EMI_per_month', 'Amount_invested_monthly', 'Monthly_Balance']
    group_col = ['Customer_ID']
    exlu_col = ['Type_of_Loan', 'Credit_History_Age']
    cat_col = df.columns.difference(num_col + drop_col + group_col + exlu_col).to_list()

    # Categorical cleaning
    cat_col.remove('Month')
    df['Credit_Mix'] = df['Credit_Mix'].replace('_', np.nan)
    df['Occupation'] = df['Occupation'].replace('_______', np.nan)
    df['Payment_Behaviour'] = df['Payment_Behaviour'].replace('!@9#%8', np.nan)
    df['Payment_of_Min_Amount'] = df['Payment_of_Min_Amount'].replace('NM', np.nan)
    # Most frequent value of the customer
    for col in cat_col:
        df[col] = df[col].fillna(
            df.groupby('Customer_ID')[col].transform(
                lambda s: s.mode().iloc[0] if not s.mode().empty else np.nan
            )
        )
    # Credit history age: "22 Years and 1 Months" -> months
    h = df['Credit_History_Age'].str.extract(r'(\d+) Years and (\d+) Months').astype(float)
    df['Credit_History_Age'] = h[0] * 12 + h[1]
    df['Credit_History_Age'] = df['Credit_History_Age'].fillna(
        df.groupby('Customer_ID')['Credit_History_Age'].transform('median'))
    num_col.append('Credit_History_Age')

    # Numeric conversion, then impossible values set to NaN
    for col in num_col:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.loc[(df['Age'] > 110) | (df['Age'] < 18), 'Age'] = np.nan
    df.loc[df['Annual_Income'] > 500000, 'Annual_Income'] = np.nan
    df.loc[df['Num_Bank_Accounts'] > 15, 'Num_Bank_Accounts'] = np.nan
    df.loc[df['Num_Credit_Card'] > 15, 'Num_Credit_Card'] = np.nan
    df.loc[df['Interest_Rate'] > 33, 'Interest_Rate'] = np.nan
    df.loc[df['Num_of_Loan'] > 9, 'Num_of_Loan'] = np.nan
    df.loc[df['Num_of_Delayed_Payment'] > 30, 'Num_of_Delayed_Payment'] = np.nan
    df.loc[df['Num_Credit_Inquiries'] > 15, 'Num_Credit_Inquiries'] = np.nan
    df.loc[df['Total_EMI_per_month'] > 500, 'Total_EMI_per_month'] = np.nan

    # Median of the customer for continuous variables
    med_col = ['Age', 'Annual_Income', 'Monthly_Inhand_Salary', 'Interest_Rate',
               'Delay_from_due_date', 'Changed_Credit_Limit',
               'Outstanding_Debt', 'Credit_Utilization_Ratio',
               'Total_EMI_per_month', 'Amount_invested_monthly', 'Monthly_Balance']
    for col in med_col:
        df[col] = df[col].fillna(df.groupby('Customer_ID')[col].transform('median'))
    # Most frequent value of the customer for counts
    freq_col = [c for c in num_col if c not in med_col]
    for col in freq_col:
        df[col] = df[col].fillna(
            df.groupby('Customer_ID')[col].transform(
                lambda s: s.mode().iloc[0] if not s.mode().empty else np.nan
            )
        )

    # One 0/1 column per type of loan
    df['Type_of_Loan'] = df['Type_of_Loan'].replace('Not Specified', np.nan)
    mode_loan = df.groupby('Customer_ID')['Type_of_Loan'].agg(
        lambda s: s.mode().iloc[0] if not s.mode().empty else np.nan)
    df['Type_of_Loan'] = df['Type_of_Loan'].fillna(df['Customer_ID'].map(mode_loan))
    s = (df['Type_of_Loan'].fillna('')
         .str.replace(' and ', ', ')
         )
    loan_lists = s.str.split(',')
    loan_lists = loan_lists.apply(lambda lst: [l.strip() for l in lst if l.strip()])
    mlb = MultiLabelBinarizer()
    loans = pd.DataFrame(mlb.fit_transform(loan_lists),
                         columns=mlb.classes_,
                         index=df.index).add_prefix('Loan_')
    df = pd.concat([df, loans], axis=1)

    return df, num_col, cat_col, list(loans.columns)
