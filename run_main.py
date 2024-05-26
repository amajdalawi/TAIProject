import sklearn
import sklearn.pipeline
from prepare_datasets import get_pruned_df, get_combined_df
from pprint import pprint
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Models to use on dataset
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor


import pandas as pd

prepared_df = get_pruned_df()
# pprint(prepared_df)
X = prepared_df.drop(columns=['real_temp'])
Y = prepared_df['real_temp']
# make test data as last week
X_train, X_test, Y_train, Y_test = train_test_split(X,Y , shuffle=False, random_state=0, test_size=1047)
# pprint(X_train)

# for future plotting and calculating predicted average arrays:
def get_temperature_df(df: pd.DataFrame = prepared_df, rows_from_last: int = -1047):
    df = df.iloc[-1047:].reset_index(drop=True)
    df = df[['real_temp','t2m']]
    original_df = get_combined_df()
    original_df = original_df[['time']].iloc[rows_from_last:].reset_index(drop=True)
    # pprint(len(original_df))
    # pprint(len(df))
    df = pd.concat([df,original_df], axis=1)
    df.to_csv("to_ignore/hmm.csv")
    # pprint(df)
    return df
    

    return df
l = get_temperature_df()

# columns in the dataframe to scale their values
columns_to_scale = ['cape','sp','tcw','sshf','slhf','msl','u10','v10','d2m','ssr','str','ttr','sund','sm','st','sd','sf','tcc','tp','mx2t6','mn2t6']

# Time-series-based cross-validator
ts_cv = TimeSeriesSplit(n_splits=20)

column_scaler = ColumnTransformer(
        transformers=[
            ("To_scale",StandardScaler(),columns_to_scale)
        ]
    )

# Linear (Ridge) Regression Pipeline 
ridge_pipeline = make_pipeline(
    column_scaler,
    GridSearchCV(estimator=Ridge(),param_grid={
        'alpha':[0,0.3,0.5,1.0,1.3,3,5]
    },scoring=["neg_mean_squared_error"],refit='neg_mean_squared_error', cv=ts_cv)
)

# K-NN Regressor Pipeline
knn_pipeline = make_pipeline(
    column_scaler,
    GridSearchCV(KNeighborsRegressor(), param_grid={
        'n_neighbors': [5,6,7,8,9,10]
    }, scoring=['neg_mean_squared_error'], refit='neg_mean_squared_error', cv=ts_cv)
)

# Multiple Layer Perceptron Regressor
mlp_pipeline = make_pipeline(
    ColumnTransformer(MinMaxScaler()),
    
)


# ridge_pipeline.fit(X_train, Y_train)
# result = ridge_pipeline.predict(X_test)

# knn_regressor = knn_pipeline.fit(X_train, Y_train)
# # pprint(knn_regressor.best_estimator)
# results = knn_regressor.predict(X_test)
# pprint(results)
# pd.DataFrame(result).to_csv("result.csv")

# knn = GridSearchCV(KNeighborsRegressor(), param_grid={
#         'n_neighbors': [5,6,7,8,9,10]
#     }, scoring=['neg_mean_squared_error'], refit='neg_mean_squared_error', cv=ts_cv).fit(X_train, Y_train)

# pprint(knn.best_params_)
# pprint(knn.best_score_)

# r = GridSearchCV(estimator=Ridge(),param_grid={
#         'alpha':[0,0.3,0.5,1.0,1.3,3,5]
#     },scoring=["neg_mean_squared_error"],refit='neg_mean_squared_error', cv=ts_cv).fit(X_train, Y_train)

# pprint(r.best_params_)
# pprint(r.best_score_)