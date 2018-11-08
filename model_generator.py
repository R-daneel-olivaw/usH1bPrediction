import data_import as di
from xgboost.sklearn import XGBRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score

def get_trained_model():
    data, X_columns, y_columns = di.prepare_data()

    X = data[X_columns]
    y = data.case_status_processed

    # my_model = XGBRegressor(n_estimators=500, n_jobs=4)
    # my_model = GradientBoostingRegressor()
    my_model = RandomForestRegressor()

    print('Scoring model ...')
    scores = cross_val_score(my_model, X, y, scoring='neg_mean_absolute_error')
    print('Sores:')
    print(scores)
    print('Mean Absolute Error %2f' % (-1 * scores.max()))

    print('Fitting model ...')
    my_model.fit(X, y)
    print('Model fitted')

    return my_model, X


