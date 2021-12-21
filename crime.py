import spark
from pyspark.sql import SparkSession
# import findspark
# findspark.init('C:\spark\spark-3.1.2-bin-hadoop3.2')
spark = SparkSession.builder.appName('Crime').getOrCreate()
sf = spark.read.csv('hdfs://localhost:6100/test.csv',
                    header=True, inferSchema=True)
sf.printSchema()

from sklearn import preprocessing
import seaborn as sns
import numpy as np
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler, LabelEncoder
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV, ElasticNetCV
from matplotlib import rcParams
import matplotlib.pyplot as plt
from collections import defaultdict
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from category_encoders import MEstimateEncoder
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_regression
from sklearn.model_selection import KFold, cross_val_score
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, roc_auc_score, accuracy_score

cat = sf.groupby("Category")["Category"].count().sort_values(ascending=False)
print("After Groupby by Category", cat)
dis = sf.groupby("PdDistrict")[
    "PdDistrict"].count().sort_values(ascending=False)
print("After Groupby District", dis)
d = defaultdict(LabelEncoder)
sf_encode = sf.apply(lambda x: d[x.name].fit_transform(x))
sf_encode = sf_encode.drop(['X', 'Y'], axis=1)

corrmat = sf_encode.corr()
f, ax = plt.subplots(figsize=(12, 12))
plot2 = sns.heatmap(corrmat, vmax=.8)
plt.xticks(rotation=90)
plt.yticks(rotation=45)
plot2.axes.set_title('Correlation Heat Map')
plt.show()

cmap1 = sns.cubehelix_palette(as_cmap=True)
k = 8
cols = corrmat.nlargest(k, 'Category')['Category'].index
cm = np.corrcoef(sf_encode[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, cmap=cmap1, square=True, annot_kws={
                 'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
hm.axes.set_title('Correlation Matrix')
plt.xticks(rotation=90)
plt.yticks(rotation=45)
plt.show()

most_dangerous_districts = sf.PdDistrict.value_counts()
_n_crime_plot = sns.barplot(
    x=most_dangerous_districts.index, y=most_dangerous_districts)
_n_crime_plot.set_xticklabels(most_dangerous_districts.index, rotation=90)

number_of_crimes = sf.Category.value_counts()

_n_crime_plot = sns.barplot(x=number_of_crimes.index, y=number_of_crimes)
_n_crime_plot.set_xticklabels(number_of_crimes.index, rotation=90)

pt = pd.pivot_table(sf, index="PdDistrict", columns="Category",
                    aggfunc=len, fill_value=0)["Dates"]
_ = pt.loc[most_dangerous_districts.index, number_of_crimes.index]
ax = sns.heatmap(_)
ax.set_title("Number of Crimes per District")


df_train  = spark.read.csv('hdfs://localhost:6100/train.csv',
                    header=True, inferSchema=True)
df_test = spark.read.csv('hdfs://localhost:6100/test.csv',
                    header=True, inferSchema=True)


def impute(df):
    s = (df.dtypes == 'object')
    object_cols = list(s[s].index)
    d = (df.dtypes != 'object')
    object_num = list(d[d].index)
    for name in object_cols:
        df[name] = df[name].fillna("none")
    for name in object_num:
        df[name] = df[name].fillna(0)
    return df


df_train[['date', 'time']] = df_train.Dates.str.split(expand=True)
df_train['Year'] = pd.DatetimeIndex(df_train['date']).year
df_train = df_train.drop(['date', 'Dates', 'Descript', 'Resolution'], axis=1)
df_train.head()
df_test[['date', 'time']] = df_test.Dates.str.split(expand=True)
df_test['Year'] = pd.DatetimeIndex(df_test['date']).year
df_test = df_test.drop(['date', 'Dates'], axis=1)
df_test.head()


# label_encoder object knows how to understand word labels.
label_encoder = preprocessing.LabelEncoder()


df_train['Category'] = label_encoder.fit_transform(df_train['Category'])
df_train['DayOfWeek'] = label_encoder.fit_transform(df_train['DayOfWeek'])
df_train['PdDistrict'] = label_encoder.fit_transform(df_train['PdDistrict'])
df_train['Address'] = label_encoder.fit_transform(df_train['Address'])
df_train['time'] = label_encoder.fit_transform(df_train['time'])

df_test['DayOfWeek'] = label_encoder.fit_transform(df_test['DayOfWeek'])
df_test['PdDistrict'] = label_encoder.fit_transform(df_test['PdDistrict'])
df_test['Address'] = label_encoder.fit_transform(df_test['Address'])
df_test['time'] = label_encoder.fit_transform(df_test['time'])

X = df_train
X = X.rename(columns={"X": "Longitude"})
X = X.rename(columns={"Y": "Latitude"})
y = X.pop("Category")

features = [
    "Longitude",
    "Latitude"
]

# Standardize
X_scaled = X.loc[:, features]
X_scaled = (X_scaled - X_scaled.mean(axis=0)) / X_scaled.std(axis=0)

kmeans = KMeans(n_clusters=10, n_init=10, random_state=0)
X["Cluster"] = kmeans.fit_predict(X_scaled)
X1 = df_test
X1 = X1.rename(columns={"X": "Longitude"})
X1 = X1.rename(columns={"Y": "Latitude"})


features = [
    "Longitude",
    "Latitude"
]

# Standardize
X_scaled_1 = X1.loc[:, features]
X_scaled_1 = (X_scaled_1 - X_scaled_1.mean(axis=0)) / X_scaled_1.std(axis=0)

kmeans = KMeans(n_clusters=10, n_init=10, random_state=0)
X1["Cluster"] = kmeans.fit_predict(X_scaled_1)

Xy = X
Xy["Cluster"] = Xy.Cluster.astype("category")
Xy["Category"] = y
sns.relplot(
    x="value", y="Category", hue="Cluster", col="variable",
    height=12, aspect=1, facet_kws={'sharex': False},
    data=Xy.melt(
        value_vars=features, id_vars=["Category", "Cluster"],
    ),
)

print(X)

X_train, X_test, y_train, y_test = train_test_split(
    X.copy(), y.copy(), test_size=0.3, random_state=0)

d_train = lgb.Dataset(X_train, label=y_train)
params = {}
params['learning_rate'] = 0.01
params['boosting_type'] = 'gbdt'  # GradientBoostingDecisionTree
params['objective'] = 'multiclass'  # Multi-class target feature
params['metric'] = 'multi_logloss'  # metric for multi-class
params['max_depth'] = 10
# no.of unique values in the target class not inclusive of the end value
params['num_class'] = 40
params['num_leaves'] = 700
clf = lgb.train(params, d_train, 100)

y_pred_1 = clf.predict(X_test)
y_pred_1 = [np.argmax(line) for line in y_pred_1]

print(accuracy_score(y_test, y_pred_1))

d_train = lgb.Dataset(X, label=y)
params = {}
params['learning_rate'] = 0.01
params['boosting_type'] = 'gbdt'  # GradientBoostingDecisionTree
params['objective'] = 'multiclass'  # Multi-class target feature
params['metric'] = 'multi_logloss'  # metric for multi-class
params['max_depth'] = 10
# no.of unique values in the target class not inclusive of the end value
params['num_class'] = 40
params['num_leaves'] = 700
clf = lgb.train(params, d_train, 100)

print(X1)

y_pred_2 = clf.predict(X1)
y_pred_2 = [np.argmax(line) for line in y_pred_2]

output = pd.DataFrame({'Id': X1.index, 'Category': y_pred_2})
print(output)

