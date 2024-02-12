import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from get_dataset import get_dataset
from descriptors import Descriptors


print('Loading data...')
X, y = get_dataset(filename='Datasets/Ivan-non-mod_3.csv', descriptors=Descriptors.pybiomed)


X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

print('Train:', len(X_train))
print('Valid:', len(X_valid), end='\n\n')

train_data = lgb.Dataset(X_train, y_train)
valid_data = lgb.Dataset(X_valid, y_valid, reference=train_data)

params = {
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'metric': {'l1', 'l2'},
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': -1
}

print('Starting training...')

# train
gbm = lgb.train(params,
                train_data,
                num_boost_round=300,
                valid_sets=[valid_data])
print()

# save model to file
print('Saving model...')
gbm.save_model('model.txt')

# predict
print('Starting predicting...', end='\n\n')
y_pred = gbm.predict(X_valid, num_iteration=gbm.best_iteration)

# eval
rmse = mean_squared_error(y_valid, y_pred) ** 0.5
mae = mean_absolute_error(y_valid, y_pred)
r2s = r2_score(y_valid, y_pred)

print('RMSE:', round(rmse, 5))
print('MAE:', round(mae, 5))
print('R2 Score:', round(r2s, 5))
