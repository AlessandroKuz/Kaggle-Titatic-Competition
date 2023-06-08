import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer
from category_encoders import TargetEncoder
from sklearn.base import BaseEstimator, TransformerMixin

from sklearn import set_config
set_config(transform_output='pandas')

df_train = pd.read_csv('./data/train.csv')
df_train.tail()
df_test = pd.read_csv('./data/test.csv')
df_test.head()

class AddTitle(TransformerMixin, BaseEstimator):
    def _init_(self):
        self.feature_names_in_ = None

    def fit(self, X, y=None):
        self.feature_names_in_ = X.columns.values
        return self

    def transform(self, X, y=None):
        X = X.copy()
        X['Title'] = X['Name'].str.findall(r',\s*([^\.]*)\s*\.').str[0]
        X['Title'] = X['Title'].apply(
            lambda x: x if x in {"Mr", "Mrs", "Miss", "Master"} else 'VIP')
        return X

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            return pd.Series(list(self.feature_names_in_) + ['Title']).astype(object)
        else:
            return pd.Series(list(input_features) + ['Title']).astype(object)


class CodifySex(TransformerMixin, BaseEstimator):
    def _init_(self):
        self.feature_names_in_ = None

    def fit(self, X, y=None):
        self.feature_names_in_ = X.columns.values
        return self

    def transform(self, X, y=None):
        X = X.copy()
        if_male = X["Sex"] == "male"
        if_female = X["Sex"] == "female"
        male_mask = X[if_male]
        female_mask = X[if_female]
        X.loc[male_mask, "Sex"] = 0
        X.loc[female_mask, "Sex"] = 1
        return X

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            return pd.Series(list(self.feature_names_in_) + ['Sex']).astype(object)
        else:
            return pd.Series(list(input_features) + ['Sex']).astype(object)


age_imputer = ColumnTransformer([
    ('age_imputer', SimpleImputer(strategy='median', missing_values=pd.NA), ['Age'])
], 
remainder='passthrough',  # passthough columns not listed in the transformers
verbose_feature_names_out=False  # set to False to get rid of the transformer's name in the output
)

# for example the function below doesn't work with a pandas Series because it expects a string
# obtain_title_transformer = FunctionTransformer(
#     lambda x: x.str.extract(r',\s*([^\.]*)\s*\.', expand=False).str.strip()
# )
# this one works with a pandas Series
obtain_title_transformer = FunctionTransformer(
    lambda x: x.iloc[:, 0].str.extract(r',\s*([^\.]*)\s*\.', expand=False).str.strip(),
    feature_names_out="one-to-one",
    validate=False
)

VIP_title_grouping = FunctionTransformer(
    lambda x: x.replace(['Dr', 'Rev', 'Col', 'Major', 'Capt', 'the Countess', 'Mlle', 'Ms', 'Lady', 'Sir', 'Mme', 'Don', 'Jonkheer'], 'VIP'),
    feature_names_out="one-to-one",
    validate=False
)

title_transformer = ColumnTransformer([
    ('title_transformer', obtain_title_transformer, ['Name']),
    ('VIP_title_grouping', VIP_title_grouping, ['Name'])
], 
remainder='passthrough',  # passthough columns not listed in the transformers
verbose_feature_names_out=False  # set to False to get rid of the transformer's name in the output
)

# replace male with 0 and female with 1
male_transformer = FunctionTransformer(
    lambda x: x.replace('male', '0')
)
female_transformer = FunctionTransformer(
    lambda x: x.replace('female', '1')
)

sex_transformer = ColumnTransformer([
    ('male_transformer', male_transformer, ["Sex"]),
    ('female_transformer', female_transformer, ["Sex"])
],
remainder='passthrough',  # passthough columns not listed in the transformers
verbose_feature_names_out=False  # set to False to get rid of the transformer's name in the output
)

drop_columns_transformer = ColumnTransformer([
    ('drop_columns_transformer', 'drop', ['SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked'])
],
remainder='passthrough',  # passthough columns not listed in the transformers
verbose_feature_names_out=False  # set to False to get rid of the transformer's name in the output
)

one_hot_encoder = ColumnTransformer([
    ('one_hot_encoder', OneHotEncoder(sparse_output=False), ["Name"], ["Age"]),
],
remainder='passthrough',  # passthough columns not listed in the transformers
verbose_feature_names_out=False  # set to False to get rid of the transformer's name in the output
)

target_encoder = ColumnTransformer([
    ('target_encoder', TargetEncoder(), ["Name"]),
],
remainder='passthrough',  # passthough columns not listed in the transformers
verbose_feature_names_out=False  # set to False to get rid of the transformer's name in the output
)

pipeline = Pipeline([
    ('age_imputer', age_imputer),
    # ('title_transformer', title_transformer),
    ('title_transformer', AddTitle()),
    # ('sex_transformer', sex_transformer),
    # ('sex_transformer', CodifySex()),
    ('drop_columns_transformer', drop_columns_transformer),
    # ('one_hot_encoder', one_hot_encoder),
    # ('target_encoder', target_encoder)
])

y_train = df_train['Survived']
X_train = df_train.drop('Survived', axis=1)
X_train.head()
print(pipeline.fit_transform(X_train, y_train))
