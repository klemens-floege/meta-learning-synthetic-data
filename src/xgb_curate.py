import numpy as np
import pandas as pd
import xgboost as xgb

from sklearn.preprocessing import StandardScaler


class CapturePredictionsCallback(xgb.callback.TrainingCallback):
    def __init__(self, dsynthetic):
        super().__init__()
        self.dsynthetic = dsynthetic
        self.predictions = []
    
    def after_iteration(self, model, epoch, evals_log):
        preds = model.predict(self.dsynthetic)
        self.predictions.append(preds)
        return False  # Return False to continue training, True would stop the training


def compute_learning_dynamics_xgb(D_train, y_train, D_synthetic, num_boost_round=10):
    # Convert datasets to DMatrix, a data structure used by XGBoost for efficiency
    dtrain = xgb.DMatrix(D_train, label=y_train)
    dsynthetic = xgb.DMatrix(D_synthetic)

    # Parameters for XGBoost - these can be tuned according to your specific problem
    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'seed': 42
    }

    # Placeholder for synthetic predictions across boosting rounds
    synthetic_predictions = []


    # Initialize the custom callback
    capture_predictions_cb = CapturePredictionsCallback(dsynthetic)

    xgb.train(params, dtrain, num_boost_round=num_boost_round, callbacks=[capture_predictions_cb])

    synthetic_predictions = np.array(capture_predictions_cb.predictions) #shape [n_checkpoints, n_samples]


    # Compute confidence and uncertainty
    confidence = synthetic_predictions.mean(axis=0) #shape [n_samples]
    aleatoric_uncertainty = np.mean(synthetic_predictions * (1 - synthetic_predictions), axis=0) #shape [n_samples]

    return confidence, aleatoric_uncertainty


#download dataset
file_path =  "../data/Adult_data/adult_dataset.csv"
data = pd.read_csv(file_path, sep=',', nrows=200)
# Split the data into features and target variable
y_train = data.iloc[:, 0].values  # Target variable is the first column
x_train = data.iloc[:, 1:].values  # Features are the rest of the columns


# Normalize the features
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)

# Import/generate synthetic data 
#TODO: insert actual synthetic data here 
d_synthetic = x_train[:50]

confidence, aleoteric_uncertainty = compute_learning_dynamics_xgb(x_train, y_train, d_synthetic, num_boost_round=20)

# Filter synthetic samples based on confidence and uncertainty thresholds
confidence_threshold = 0.5
uncertainty_threshold = 0.2

# Apply thresholds to filter indices
filtered_indices = (confidence > confidence_threshold) & (aleoteric_uncertainty < uncertainty_threshold)
d_filtered = d_synthetic[filtered_indices]

print(f"Original number of synthetic samples: {d_synthetic.shape[0]}")
print(f"Number of filtered synthetic samples: {d_filtered.shape[0]}")
