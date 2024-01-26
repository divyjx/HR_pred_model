import sys
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from joblib import dump, load
from lightgbm import LGBMRegressor

def check_dir(name):
    if not os.path.exists(name):
        print(f"{name} file or directory is not present, run train.py to build model first")
        exit()

def pre_process_test(data):

  # loding encoders
  oh_encoder = load("models/oh_encoder.bin")
  std_scaler = load("models/std_scaler.bin")

  data = data.drop(['datasetId'], axis = 1)
  condition_col = data.pop('condition')
  uuid_col = data.pop('uuid')

  columns = data.columns
  scaled_data = std_scaler.transform(data.to_numpy())
  scaled_data = pd.DataFrame(scaled_data, columns = columns)

  condition_col = np.array(condition_col.values).reshape(-1, 1)
  enc_col = oh_encoder.transform(condition_col)
  scaled_data["is_interruption"] = enc_col[:, 0]
  scaled_data["is_no_stress"] = enc_col[:, 1]
  scaled_data["is_time_pressure"] = enc_col[:, 2]

  return scaled_data, uuid_col

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

def inference():
    test_file = sys.argv[1]
    check_dir('models')
    check_dir(test_file)

    test_data = pd.read_csv(test_file)
    test_feat, uuid_col = pre_process_test(test_data)

    test_feat = extract_features(test_feat)

    model = load('models/model.bin')
    print("Model params :\n" + str(model.get_params()))
    predictions = model.predict(test_feat) 

    result = pd.DataFrame()
    result['uuid'] = uuid_col.values
    result['HR'] = predictions
    result.set_index('uuid', inplace=True)
    result.to_csv('results.csv', index = 'uuid')

    print("\nResults are saved in 'results.csv'")


if __name__=='__main__':
    if (len(sys.argv)!=2):
        print('usage: python run.py <test_file.csv>')
        exit()
    inference()
    