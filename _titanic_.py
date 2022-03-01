import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report, plot_roc_curve
from sklearn.model_selection import cross_validate
from sklearn.preprocessing import LabelEncoder, RobustScaler

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)

#Identification of important functions.They will be used when needed.

def outlier_thresholds(dataframe, col_name, q1=0.05, q3=0.95):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

def grab_col_names(dataframe, cat_th=10, car_th=20):
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')
    return cat_cols, num_cols, cat_but_car

def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False

def grab_outliers(dataframe, col_name, index=False):
    low, up = outlier_thresholds(dataframe, col_name)

    if dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].shape[0] > 10:
        print("#####################################################")
        print(str(col_name) + " variable have too much outliers: " + str(dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].shape[0]))
        print("#####################################################")
        print(dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].head(15))
        print("#####################################################")
        print("Lower threshold: " + str(low) + "   Lowest outlier: " + str(dataframe[col_name].min()) +
              "   Upper threshold: " + str(up) + "   Highest outlier: " + str(dataframe[col_name].max()))
        print("#####################################################")
    elif (dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].shape[0] < 10) & \
            (dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].shape[0] > 0):
        print("#####################################################")
        print(str(col_name) + " variable have less than 10 outlier values: " + str(dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].shape[0]))
        print("#####################################################")
        print(dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))])
        print("#####################################################")
        print("Lower threshold: " + str(low) + "   Lowest outlier: " + str(dataframe[col_name].min()) +
              "   Upper threshold: " + str(up) + "   Highest outlier: " + str(dataframe[col_name].max()))
        print("#####################################################")
    else:
        print("#####################################################")
        print(str(col_name) + " variable does not have outlier values")
        print("#####################################################")

    if index:
        print(str(col_name) + " variable's outlier indexes")
        print("#####################################################")
        outlier_index = dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].index
        return outlier_index

def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

def missing_values_table(dataframe):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]

    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    if len(missing_df) > 0:
        print(missing_df, end="\n")
    else:
        print("No NaN values")

def rare_analyser(dataframe, target, cat_cols):
    for col in cat_cols:
        print(col, ":", len(dataframe[col].value_counts()))
        print(pd.DataFrame({"COUNT": dataframe[col].value_counts(),
                            "RATIO": dataframe[col].value_counts() / len(dataframe),
                            "TARGET_MEAN": dataframe.groupby(col)[target].mean()}), end="\n\n\n")

def one_hot_encoder(dataframe, categorical_cols, drop_first=True):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe

def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe

def rare_encoder(dataframe, rare_perc):
    temp_df = dataframe.copy()

    rare_columns = [col for col in temp_df.columns if temp_df[col].dtypes == 'O'
                    and (temp_df[col].value_counts() / len(temp_df) < rare_perc).any(axis=None)]

    for var in rare_columns:
        tmp = temp_df[var].value_counts() / len(temp_df)
        rare_labels = tmp[tmp < rare_perc].index
        temp_df[var] = np.where(temp_df[var].isin(rare_labels), 'Rare', temp_df[var])

    return temp_df


df_ = pd.read_csv(r"C:\Users\hp\PycharmProjects\pythonProject2\titanic (1).csv")
df = df_.copy()
df.head(10)
df.columns = [col.upper() for col in df.columns]

missing_values_table(df)

#Feature Engineering Part

df["NEW_CABIN_BOOL"] = df["CABIN"].notnull().astype('int')

del df["CABIN"]

df['NEW_TITLE'] = df.NAME.str.extract('([A-Za-z]+)\.', expand=False)

df["AGE"] = df["AGE"].fillna(df.groupby("NEW_TITLE")["AGE"].transform("median"))

df["NEW_FAMILY_SIZE"] = df["SIBSP"] + df["PARCH"] + 1

df["NEW_AGE_PCLASS"] = df["AGE"] * df["PCLASS"]

df.loc[((df['SIBSP'] + df['PARCH']) > 0), "NEW_IS_ALONE"] = "NO"
df.loc[((df['SIBSP'] + df['PARCH']) == 0), "NEW_IS_ALONE"] = "YES"

df.loc[(df['AGE'] < 18), 'NEW_AGE_CAT'] = 'young'
df.loc[(df['AGE'] >= 18) & (df['AGE'] < 56), 'NEW_AGE_CAT'] = 'mature'
df.loc[(df['AGE'] >= 56), 'NEW_AGE_CAT'] = 'senior'

df.loc[(df['SEX'] == 'male') & (df['AGE'] <= 21), 'NEW_SEX_CAT'] = 'youngmale'
df.loc[(df['SEX'] == 'male') & ((df['AGE'] > 21) & (df['AGE']) <= 50), 'NEW_SEX_CAT'] = 'maturemale'
df.loc[(df['SEX'] == 'male') & (df['AGE'] > 50), 'NEW_SEX_CAT'] = 'seniormale'
df.loc[(df['SEX'] == 'female') & (df['AGE'] <= 21), 'NEW_SEX_CAT'] = 'youngfemale'
df.loc[(df['SEX'] == 'female') & ((df['AGE'] > 21) & (df['AGE']) <= 50), 'NEW_SEX_CAT'] = 'maturefemale'
df.loc[(df['SEX'] == 'female') & (df['AGE'] > 50), 'NEW_SEX_CAT'] = 'seniorfemale'

df["NEW_AGE_PCLASS_FARE"] = df["NEW_AGE_PCLASS"] / df["FARE"]
df['NEW_AGE_PCLASS_FARE'].replace([np.inf], '0', inplace=True)
df["NEW_AGE_PCLASS_FARE"] = df["NEW_AGE_PCLASS_FARE"].astype(float)


# Implementing mode to the NaN values of categoric variables.
df = df.apply(lambda x: x.fillna(x.mode()[0]) if (x.dtype == "O" and len(x.unique()) <= 10) else x, axis=0)


cat_cols, num_cols, cat_but_car = grab_col_names(df)

num_cols = [col for col in num_cols if "PASSENGERID" not in col]

for col in num_cols:
    print(col, check_outlier(df, col))

for col in num_cols:
    grab_outliers(df, col)

for col in num_cols:
    replace_with_thresholds(df, col)

remove_cols = ["TICKET", "NAME"]
df.drop(remove_cols, inplace=True, axis=1)

# Label Encoding for columns that aren't integer or float and have nunique = 2

binary_cols = [col for col in df.columns if df[col].dtype not in [int, float]
               and df[col].nunique() == 2]

for col in binary_cols:
    df = label_encoder(df, col)

# Rare analyser for dataset:
rare_analyser(df, "SURVIVED", cat_cols)
df = rare_encoder(df, 0.01)

# One-Hot Encoding for variables with levels not-comparable with each other:
ohe_cols = [col for col in df.columns if 10 >= df[col].nunique() > 2]
df = one_hot_encoder(df, ohe_cols)

#We must check columns again.
cat_cols, num_cols, cat_but_car = grab_col_names(df)

num_cols = [col for col in num_cols if "PASSENGERID" not in col]

#Dropping useless columns.
useless_cols = [col for col in df.columns if df[col].nunique() == 2 and
                (df[col].value_counts() / len(df) < 0.01).any(axis=None)]

useless_cols

df.drop(useless_cols, axis=1, inplace=True)


#Scaling Part
rs = RobustScaler()
df[num_cols] = rs.fit_transform(df[num_cols])

#let's move on to the model part

y = df["SURVIVED"]
X = df.drop(["PASSENGERID", "SURVIVED"], axis=1)

# Model
log_model = LogisticRegression().fit(X, y)

# b - bias value:

log_model.intercept_[0]

# w - weight values:

log_model.coef_[0]

# Prediction of the dependent column:

y_pred = log_model.predict(X)

def plot_confusion_matrix(y, y_pred):
    acc = round(accuracy_score(y, y_pred), 2)
    cm = confusion_matrix(y, y_pred)
    sns.heatmap(cm, annot=True, fmt=".0f", cmap="GnBu")
    plt.xlabel('y_pred')
    plt.ylabel('y')
    plt.title('Accuracy Score: {0}'.format(acc), size=10)
    plt.show()


plot_confusion_matrix(y, y_pred)

print(classification_report(y, y_pred))

y_prob = log_model.predict_proba(X)[:, 1]
roc_auc_score(y, y_prob)


plot_roc_curve(log_model, X, y, color="k")
plt.title('ROC Curve')
plt.plot([0, 1], [0, 1], color="b", marker="x")
plt.show()

#We built the model using the whole dataset. It looks very successful according to the metrics.
# But we need to do model validation and then we can comment.

# Model Validation:'10-Fold Cross Validation'

y = df["SURVIVED"]
X = df.drop(["PASSENGERID", "SURVIVED"], axis=1)
log_model = LogisticRegression().fit(X, y)


cv_results = cross_validate(log_model,
                            X, y,
                            cv=10,
                            scoring=["accuracy", "precision", "recall", "f1", "roc_auc"])

def cv_mean(cv_nums):
    print(f"Test Accuracy: {cv_nums['test_accuracy'].mean():.4f}")
    print(f"Test Precision: {cv_nums['test_precision'].mean():.4f}")
    print(f"Test Recall: {cv_nums['test_recall'].mean():.4f}")
    print(f"Test F1: {cv_nums['test_f1'].mean():.4f}")
    print(f"Test ROC AUC: {cv_results['test_roc_auc'].mean():.4f}")

cv_mean(cv_results)


#At the end of the 'validation' we can say that these results are not better than the previous ones but they are more reliable and fine.

