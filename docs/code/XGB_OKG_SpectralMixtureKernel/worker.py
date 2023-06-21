# worker.py

import pickle
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import concurrent.futures

# Load the KFold object
with open('kf.pkl', 'rb') as f:
    kf = pickle.load(f)
 # Load data
with open('xgb_params.pkl', 'rb') as f:
    xgb_params = pickle.load(f)

X = np.load("X.npy")
y = np.load("y.npy")
des_transformed = np.load("des_transformed.npy",allow_pickle=True)
des_transformed = des_transformed.astype("object")
# Convert 'num_boost_round' column to the nearest integer

num_boost_round_index = list(xgb_params.keys()).index('num_boost_round')
des_transformed[:, num_boost_round_index] = [int(round(float(x))) for x in des_transformed[:, num_boost_round_index]]

# Convert 'max_depth' column to the nearest integer
max_depth_index = list(xgb_params.keys()).index('max_depth')
des_transformed[:, max_depth_index] = [int(round(float(x))) for x in des_transformed[:, max_depth_index]]


def process_hyperparams(args):
    hyperparams, (train_index, test_index) = args
    try:
        # Create a dictionary from the hyperparameters
        hyperparams_dict = {key: value for key, value in zip(xgb_params.keys(), hyperparams)}

        class XGBEstimator:
            def __init__(self, **kwargs):
                self.params = kwargs

            def fit(self, X, y):
                dtrain = xgb.DMatrix(X, label=y)
                self.model = xgb.train(params=self.params, dtrain=dtrain, num_boost_round=self.params.pop('num_boost_round'))
                return self

            def predict(self, X):
                dtest = xgb.DMatrix(X)
                return self.model.predict(dtest)

        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model', XGBEstimator(**hyperparams_dict))
        ])

        # Split the data into training and test sets
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # Train the model and calculate the test error
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        mse = np.mean((y_pred - y_test)**2)
        hyperparams_dict['mse'] = mse
        return hyperparams_dict
    except Exception as e:
        print(f"Error encountered: {e}")
        raise

def main(X, y, kf, xgb_params, des_transformed):
    with concurrent.futures.ProcessPoolExecutor() as executor:
        # Create a list of tuples where each tuple is (hyperparams, (train_index, test_index))
        inputs = [(hyperparams, split) for hyperparams in des_transformed for split in kf.split(X, y)]
        results = list(executor.map(process_hyperparams, inputs))
        return results

if __name__ == '__main__':
    import time
    start_time = time.time()
    try:
        results = main(X, y, kf, xgb_params, des_transformed)
    except Exception as e:
        print("An error occurred:", e)
        raise

    end_time = time.time()  # capture end time
    print(f"Total runtime: {end_time - start_time} seconds")

    # Convert the results to a pandas dataframe
    tested_hypers = pd.DataFrame(results)

    # Save the results to a file
    tested_hypers.to_csv("results.csv",index=False)

