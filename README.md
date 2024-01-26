# Heart Rate Prediction Model 🔥
A powerful Machine Learning model designed to predict heart rates, developed for the AI Cure: Parsec 4.0 competition.

### Data Preparation
Data preprocessing was a crucial step, involving the removal of redundant features, scaling the dataset, and separating labels for better model training. To gain more insights into the dataset, we utilized visualization techniques. This includes histogram plots of feature vectors and a correlation heatmap. Then, highly correlated features were removed, resulting in a reduction of features from 36 to 32 without compromising performance. Some features were averaged out, while others were directly removed.

### Model Training
Multiple regression models were considered during the training phase, including Linear Regression (as a benchmark), RandomForestRegressor, XGBRegressor, and LGBMRegressor. We used Mean Squared Error (MSE) as our evaluation metric.  Initially, they were trained on 80% of the data and evaluated on the remaining 20%. Further evaluation utilized 5-fold cross-validation, revealing the superior performance of the LGBMRegressor. 

### Final Model
The LGBMRegressor model was hyperparameter-tuned using random grid search to identify optimal parameters, ensuring the lowest MSE in heart rate predictions.

Feel free to explore the '**training.ipynb**' notebook for a detailed walkthrough of the training process and results. 

# Setup 🛠️
Clone this repository
```bash
git clone https://github.com/divyjx/HR_pred_model.git
cd HR_pred_model
```
Create python virtual enviornment using python 3.10.12 and clone PS repository
```bash
python -m venv ./myenv
source myenv/bin/activate
pip install -r requirements.txt
git clone https://github.com/sriharikgiitdh/AI_CURE_PARSEC.git
```

# Testing 🧪
Execute the command below to generate a 'results.csv' file using the trained model.
```bash
python run.py test_data.csv
```

# Training 🚀
This script trains the model on data from 'train_data.csv' in the 'AI_CURE_PARSEC' folder. After training, it saves the model and encoders in the 'models' folder for future inference.
```bash
python train.py 
```
For comprehensive training steps, experiments, and visualizations, please refer to the '**training.ipynb**' notebook. Ensure that you choose the appropriate Python kernel for executing the notebook.

# Rsults
![Result on test split(20%)](https://github.com/divyjx/HR_pred_model/blob/main/images/test_80_20.png?raw=true)
![Results of 5-Fold Cross Validation](https://github.com/divyjx/HR_pred_model/blob/main/images/test_5_fold.png?raw=true)

Optimized LGBMRegressor, as the final model, gave average Mean Squared Error (MSE) of **0.1310** in 5-fold cross-validation.

# Team Info 
Team Name : TM 
<table>
  <tr>
    <th>Members</th>
    <th>Contact</th>
    <th>Roll Number</th>
  </tr>
  
  <tr>
    <td>Divy jain</td>
    <td>9516644309</td>
    <td>210010015</td>
  </tr>
  
  <tr>
    <td>Karthik Hegde</td>
    <td>--</td>
    <td>210010022</td>
  </tr>
</table>


