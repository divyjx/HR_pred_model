# HR_pred_model 🔥
A Machine Learning model for predicting the Heart Rate. \
Competition - AI Cure: Parsec 4.0


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

# Team Info 🤝
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


