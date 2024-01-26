import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from joblib import dump, load
from lightgbm import LGBMRegressor
from sklearn.model_selection import cross_val_score


def check_dir():
    model_directory = "./models"
    if not os.path.exists(model_directory):
        os.makedirs(model_directory)

def pre_process_train(data):

  oh_encoder = OneHotEncoder(sparse_output = False)
  std_scaler = StandardScaler()

  data = data.drop(['datasetId'], axis = 1)
  condition_col = data.pop('condition')
  uuid_col = data.pop('uuid')
  labels = data.pop('HR')

  columns = data.columns
  scaled_data = std_scaler.fit_transform(data.to_numpy())
  scaled_data = pd.DataFrame(scaled_data, columns = columns)

  condition_col = np.array(condition_col.values).reshape(-1, 1)
  enc_col = oh_encoder.fit_transform(condition_col)
  scaled_data["is_interruption"] = enc_col[:, 0]
  scaled_data["is_no_stress"] = enc_col[:, 1]
  scaled_data["is_time_pressure"] = enc_col[:, 2]

  scaled_data["HR"] = labels.values

  # saving encoders
  dump(oh_encoder, 'models/oh_encoder.bin', compress=True)
  dump(std_scaler, 'models/std_scaler.bin', compress=True)

  return scaled_data

def extract_features(data):

  data['AVG2'] = data[['TP', 'VLF']].mean(axis=1)
  data['AVG3'] = data[['HF_LF', 'HF_NU']].mean(axis=1)

  data = data.drop([
      'KURT_REL_RR',
      'SKEW_REL_RR',
      'SDSD_REL_RR',
      'SDSD',
      'HF_LF',
      'HF_NU',
  ], axis=1)

  return data

def train():
    
    check_dir()

    # pre-process
    train_data = pd.read_csv("AI_CURE_PARSEC/train_data.csv")
    preprocessed_data = pre_process_train(train_data)
    train_feat, train_labels = preprocessed_data.drop(['HR'], axis = 1), preprocessed_data['HR']

    # extract features
    train_feat = extract_features(train_feat)
    X, y = train_feat.values, train_labels.values

    # model eval
    best_parameters = {'learning_rate': 0.05958008171890075,
                        'max_depth': 29, 'n_estimators': 376,
                        'num_leaves': 28}
    model = LGBMRegressor(force_col_wise=True, verbose = 0, **best_parameters)     # 0.1310 - BEST
    scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error')
    average_mse = -1 * scores.mean()
    print('LGBMRegressor(random search) Average MSE: {:.4f}'.format(average_mse))

    # final model
    best_model = LGBMRegressor(
        force_col_wise=True,
        verbose = 0,
        **best_parameters)

    best_model.fit(X, y)
    dump(best_model, "models/model.bin")

    print('Trained model and enocoders are saved in models/')
    
if __name__ == "__main__":
    train()