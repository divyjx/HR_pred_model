import sys
import pandas as pd
from sklearn.metrics import mean_squared_error

try:
    predictions_df = pd.read_csv(sys.argv[1])
    truth_df = pd.read_csv(sys.argv[2])
except Exception as e:
    print(e)
    print("\nUsage : python eval.py results.csv truth.csv")
    exit()

merged_df = pd.merge(predictions_df, truth_df, on='uuid', suffixes=('_pred', '_true'))

# Calculate MSE
mse = mean_squared_error(merged_df['HR_pred'], merged_df['HR_true'])
print(f"Mean Squared Error (MSE) of HR: {mse}")
