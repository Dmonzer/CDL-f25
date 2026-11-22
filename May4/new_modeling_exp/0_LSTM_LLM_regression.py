#!/usr/bin/env python
# coding: utf-8

# In[10]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas import read_csv
import math
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.layers import Dense, Activation, Dropout
import time #helper libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas import read_csv
import math
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.layers import Dense, Activation, Dropout
import time #helper libraries
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
import itertools
from sklearn.metrics import mean_absolute_error, r2_score, mean_absolute_percentage_error
import os
import keras_tuner as kt
from tensorflow.keras.optimizers import Adam
import os

#import tensorflow as tf
#print("GPUs available:", tf.config.list_physical_devices('GPU'))

'''
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
'''
# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[11]:


#df= pd.read_csv("processed_news_scores_with_stocks_WITH_AGGREGATE_ready_formodeling.csv")
#score_cols = [f"scores_{i}" for i in range(23)]
#df = df.drop(columns=score_cols)
df_tesla = pd.read_csv("Tesla_processed_news_scores_with_stocks_WITH_AGGREGATE_ready_formodeling.csv")
df_apple= pd.read_csv("Apple_processed_news_scores_with_stocks_WITH_AGGREGATE_ready_formodeling.csv")
df_amazon= pd.read_csv("Amazon_processed_news_scores_with_stocks_WITH_AGGREGATE_ready_formodeling.csv")


# In[12]:


df_tesla = df_tesla.sort_values(by="timestamp").reset_index(drop=True)
df_apple = df_apple.sort_values(by="timestamp").reset_index(drop=True)
df_amazon = df_amazon.sort_values(by="timestamp").reset_index(drop=True)


# In[13]:


df_tesla.isna().any()


# # LSTM

# ## Close price with avg min, and max score of all. 

# In[14]:


# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=1, target_col=0):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back), :]          # full feature window
        dataX.append(a)
        dataY.append(dataset[i + look_back, target_col])
    return np.array(dataX), np.array(dataY)




# #### Changes implemented on nov 30 code:
# 1- Adding stopping epochs. 
# 
# 2- add validation dataset
# 
# 3- SPLIT then normalize (it intentionally uses train's min/max for everything, including test. Here's why:
# In real life, when you deploy a model, you don't know the future's min/max. So you scale future data using what you learned from the past (train). It's the only honest approach.
# If test had different min/max than train, that's fine — the scaler still applies train's min/max to test. The test values might go slightly outside [0,1] range, but that's acceptable and actually realistic.)
# 
# 4- Adding evaluation metrics 
# 
# Limitations to note:
# - the time series, does not have same interval gaps due to market closure, for example some datapoints might have 1 hour gaps while others have 9. News are accumulated within the 9 hours of market closure and treated equally. otherwise the datapoint gaps are treated similarly. 

# In[15]:


#######################NEW###########################
from tensorflow.keras.callbacks import EarlyStopping

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=20,        # stops if no improvement for 20 epochs
    restore_best_weights=True
)
#######################NEW###########################


# In[ ]:


n_repeats = 3 #10
lookbacks = np.arange(15, 100, 15)
train_sizes = [0.7] #[0.6, 0.7, 0.8]
rows = []

# NEW: dictionary mapping company -> dataframe
companies = {
    "Tesla":  df_tesla,
    "Apple":  df_apple,
    "Amazon": df_amazon,
}

# NEW: shared columns since you renamed them


# ===== HYPERTUNING SECTION (regression multivariate) =====
cols = ["close", "avg_norm_avg_scores", "avg_norm_max_scores", "avg_norm_min_scores"]
TUNE_LOOKBACK = 30
TUNE_TS = 0.7

best_hyperparams_per_company = {}

for company, df in companies.items():
    print(f"\n=== Hypertuning {company} (regression multivariate) ===")

    dataset = df[cols].values
    target_col = 0

    train_size = int(len(dataset) * TUNE_TS)
    train, test = dataset[0:train_size, :], dataset[train_size:, :]

    scaler = MinMaxScaler(feature_range=(0, 1))
    train = scaler.fit_transform(train)
    test = scaler.transform(test)

    trainX, trainY = create_dataset(train, TUNE_LOOKBACK, target_col)
    n_features = dataset.shape[1]
    trainX = trainX.reshape(trainX.shape[0], TUNE_LOOKBACK, n_features)

    def build_model(hp):
        model = Sequential()
        n_layers = hp.Int("n_layers", 1, 2)
        units    = hp.Choice("units", [25, 50, 100])
        dropout  = hp.Choice("dropout", [0.1, 0.2, 0.3])
        lr       = hp.Choice("lr", [1e-3, 1e-2])

        if n_layers == 1:
            model.add(LSTM(units, input_shape=(TUNE_LOOKBACK, n_features)))
            model.add(Dropout(dropout))
        else:
            model.add(LSTM(units, return_sequences=True, input_shape=(TUNE_LOOKBACK, n_features)))
            model.add(Dropout(dropout))
            model.add(LSTM(units))
            model.add(Dropout(dropout))

        model.add(Dense(1))
        model.compile(loss="mse", optimizer=Adam(learning_rate=lr))
        return model

    tuner = kt.BayesianOptimization(
        build_model,
        objective="val_loss",
        max_trials=15,
        directory="nov30_codes_new_data/LSTM-HYPERTUNING",
        project_name=f"reg_multivariate_{company}",
        overwrite=True,
    )

    tuner.search(
        trainX, trainY,
        epochs=50,
        validation_split=0.1,
        batch_size=240,
        callbacks=[early_stop],
        verbose=0,
    )

    best_hp = tuner.get_best_hyperparameters(1)[0]
    best_hyperparams_per_company[company] = {
        "n_layers": best_hp.get("n_layers"),
        "units":    best_hp.get("units"),
        "dropout":  best_hp.get("dropout"),
        "lr":       best_hp.get("lr"),
    }
    print(f"Best for {company}: {best_hyperparams_per_company[company]}")

    #################################
    best_mse = float("inf")
    worst_mse = -float("inf")

    for lookback, ts in itertools.product(lookbacks, train_sizes):
        for repeat in range(n_repeats):

            np.random.seed(5)
            dataset = df[cols].values
            target_col = 0

            train_size = int(len(dataset) * ts)
            train, test = dataset[0:train_size, :], dataset[train_size:, :]

            scaler = MinMaxScaler(feature_range=(0, 1))
            train = scaler.fit_transform(train)
            test = scaler.transform(test)
            dataset = np.concatenate([train, test], axis=0)

            look_back = lookback
            trainX, trainY = create_dataset(train, look_back, target_col)
            testX, testY = create_dataset(test, look_back, target_col)

            n_features = dataset.shape[1]
            trainX = trainX.reshape(trainX.shape[0], look_back, n_features)
            testX = testX.reshape(testX.shape[0], look_back, n_features)
            '''
            print("trainX NaNs:", np.isnan(trainX).sum())
            print("testX NaNs:", np.isnan(testX).sum())
            print("trainY NaNs:", np.isnan(trainY).sum())
            print("testY NaNs:", np.isnan(testY).sum())
            '''
            hp = best_hyperparams_per_company[company]
            model = Sequential()
            if hp["n_layers"] == 1:
                model.add(LSTM(hp["units"], input_shape=(look_back, n_features)))
                model.add(Dropout(hp["dropout"]))
            else:
                model.add(LSTM(hp["units"], return_sequences=True, input_shape=(look_back, n_features)))
                model.add(Dropout(hp["dropout"]))
                model.add(LSTM(hp["units"]))
                model.add(Dropout(hp["dropout"]))
            model.add(Dense(1))
            model.compile(loss="mse", optimizer=Adam(learning_rate=hp["lr"]))
            
            
            model.fit(trainX, trainY,
                      epochs=1000,
                      batch_size=240,
                      verbose=0,
                      validation_split=0.1,
                      callbacks=[early_stop])

            trainPredict = model.predict(trainX)
            testPredict = model.predict(testX)
            '''
            print("raw trainPredict NaNs:", np.isnan(trainPredict).sum())
            print("raw testPredict NaNs:", np.isnan(testPredict).sum())
            '''
            trainPredict = scaler.inverse_transform(np.repeat(trainPredict, n_features, axis=1))[:,target_col]
            trainY_inv = scaler.inverse_transform(np.repeat(trainY.reshape(-1,1), n_features, axis=1))[:,target_col]
            testPredict = scaler.inverse_transform(np.repeat(testPredict, n_features, axis=1))[:,target_col]
            testY_inv = scaler.inverse_transform(np.repeat(testY.reshape(-1,1), n_features, axis=1))[:,target_col]
            '''
            print("trainPredict NaNs:", np.isnan(trainPredict).sum())
            print("trainY_inv NaNs:", np.isnan(trainY_inv).sum())
            print("testPredict NaNs:", np.isnan(testPredict).sum())
            print("testY_inv NaNs:", np.isnan(testY_inv).sum())
            '''
            trainScore = math.sqrt(mean_squared_error(trainY_inv, trainPredict))
            testScore = math.sqrt(mean_squared_error(testY_inv, testPredict))
            # Additional metrics
            train_mae  = mean_absolute_error(trainY_inv, trainPredict)
            test_mae   = mean_absolute_error(testY_inv, testPredict)
            train_mape = mean_absolute_percentage_error(trainY_inv, trainPredict)
            test_mape  = mean_absolute_percentage_error(testY_inv, testPredict)
            train_r2   = r2_score(trainY_inv, trainPredict)
            test_r2    = r2_score(testY_inv, testPredict)

            # Directional accuracy (% of times predicted up/down direction matches actual)
            train_dir_acc = np.mean(np.sign(np.diff(trainY_inv)) == np.sign(np.diff(trainPredict)))
            test_dir_acc  = np.mean(np.sign(np.diff(testY_inv))  == np.sign(np.diff(testPredict)))

            dataset_target = scaler.inverse_transform(dataset)[:, target_col]

            trainPredictPlot = np.empty_like(dataset_target)
            trainPredictPlot[:] = np.nan
            trainPredictPlot[look_back:len(trainPredict)+look_back] = trainPredict

            testPredictPlot = np.empty_like(dataset_target)
            testPredictPlot[:] = np.nan
            testPredictPlot[len(trainPredict)+(look_back*2)+1 : len(dataset_target)-1] = testPredict

            plt.plot(dataset_target)
            plt.plot(trainPredictPlot)
            plt.plot(testPredictPlot)

            testPrices = dataset_target[len(dataset_target)-len(testPredict):]

            trainScore = math.sqrt(mean_squared_error(trainY_inv, trainPredict))
            testScore = math.sqrt(mean_squared_error(testY_inv, testPredict))

            if testScore < best_mse:
                best_mse = testScore
                plt.figure(figsize=(12,6))
                plt.plot(dataset_target, label="Actual Prices (Full Dataset)", color="blue")
                plt.plot(trainPredictPlot, label="Train Predictions", color="orange")
                plt.plot(testPredictPlot, label="Test Predictions", color="green")
                plt.legend()
                plt.title(f"LSTM-LLM Predictions vs Actual Prices - {company}")
                plt.xlabel("Time Steps")
                plt.ylabel("Price")
                plt.grid(True)
                #plt.savefig(f"nov30_codes_new_data/LSTM-HYPERTUNING/hypertuned_LSTM_LLM_Regression_best_mse_{company}_lb{lookback}_ts{ts}_repeat{repeat}.png")
                testPredictPlot=pd.Series(testPredictPlot).dropna().values
                test_size = testPredictPlot.shape[0]
                plt.figure(figsize=(12,6))
                plt.plot(dataset_target[-test_size:], label="Actual Prices (Full Dataset)", color="blue")
                plt.plot(testPredictPlot, label="Test Predictions", color="green")
                plt.legend()
                plt.title(f"LSTM-LLM Predictions vs Actual Prices - {company}")
                plt.xlabel("Time Steps")
                plt.ylabel("Price")
                plt.grid(True)
                plt.savefig(f"nov30_codes_new_data/LSTM-HYPERTUNING/hypertuned_LSTM_LLM_Regression_best_mse_test_only_{company}_lb{lookback}_ts{ts}_repeat{repeat}.png")

            if testScore > worst_mse:
                worst_mse = testScore
                plt.figure(figsize=(12,6))
                plt.plot(dataset_target, label="Actual Prices (Full Dataset)", color="blue")
                plt.plot(trainPredictPlot, label="Train Predictions", color="orange")
                plt.plot(testPredictPlot, label="Test Predictions", color="green")
                plt.legend()
                plt.title(f"LSTM-LLM Predictions vs Actual Prices - {company}")
                plt.xlabel("Time Steps")
                plt.ylabel("Price")
                plt.grid(True)
                #plt.savefig(f"nov30_codes_new_data/LSTM-HYPERTUNING/hypertuned_LSTM_LLM_Regression_worst_mse_{company}_lb{lookback}_ts{ts}_repeat{repeat}.png")
                testPredictPlot=pd.Series(testPredictPlot).dropna().values
                test_size = testPredictPlot.shape[0]
                plt.figure(figsize=(12,6))
                plt.plot(dataset_target[-test_size:], label="Actual Prices (Full Dataset)", color="blue")
                plt.plot(testPredictPlot, label="Test Predictions", color="green")
                plt.legend()
                plt.title(f"LSTM-LLM Predictions vs Actual Prices - {company}")
                plt.xlabel("Time Steps")
                plt.ylabel("Price")
                plt.grid(True)
                plt.savefig(f"nov30_codes_new_data/LSTM-HYPERTUNING/hypertuned_LSTM_LLM_Regression_worst_mse_test_only_{company}_lb{lookback}_ts{ts}_repeat{repeat}.png")

            #result = {"company": company, "n_repeat": repeat, "lookback": look_back, "test_size": 1-ts, "train_rmse": trainScore, "test_rmse": testScore}
            result = {
                "company": company,
                "n_repeat": repeat,
                "lookback": look_back,
                "test_size": 1-ts,
                "train_rmse": trainScore,
                "test_rmse": testScore,
                "train_mae": train_mae,
                "test_mae": test_mae,
                "train_mape": train_mape,
                "test_mape": test_mape,
                "train_r2": train_r2,
                "test_r2": test_r2,
                "train_dir_acc": train_dir_acc,
                "test_dir_acc": test_dir_acc,
            }
            rows.append(result)

df_results = pd.DataFrame(rows)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


df_results.to_csv("nov30_codes_new_data/LSTM-HYPERTUNING/hypertuned_LSTM_LLM_Regression_v2.csv", index=False)


# In[ ]:


# ===== Summary plot: metric distributions per company (regression multivariate) =====
import matplotlib.pyplot as plt
import numpy as np

MODEL_NAME = "LSTM-LLM Regression"
SAVE_DIR   = "nov30_codes_new_data/LSTM-HYPERTUNING"

metrics = ["test_rmse", "test_mae", "test_mape", "test_r2", "test_dir_acc"]
metric_labels = ["RMSE", "MAE", "MAPE", "R²", "Directional Acc."]
bounded_01 = {"test_dir_acc"}

company_order = ["Tesla", "Apple", "Amazon"]
company_colors = {"Tesla": "#E63946", "Apple": "#457B9D", "Amazon": "#2A9D8F"}

fig, axes = plt.subplots(1, len(metrics), figsize=(4 * len(metrics), 5), sharex=True)
if len(metrics) == 1:
    axes = [axes]

for ax, m, lbl in zip(axes, metrics, metric_labels):
    for company in company_order:
        sub = df_results[df_results["company"] == company]
        if sub.empty:
            continue

        agg = sub.groupby("lookback")[m].agg(["mean", "std", "count"]).reset_index()
        agg["std"] = agg["std"].fillna(0)

        color = company_colors[company]

        jitter = (np.random.RandomState(0).rand(len(sub)) - 0.5) * 1.5
        ax.scatter(sub["lookback"].values + jitter, sub[m].values,
                   color=color, alpha=0.35, s=25, edgecolor="none")

        ax.plot(agg["lookback"], agg["mean"], color=color, lw=2, label=company, marker="o")
        ax.fill_between(agg["lookback"],
                        agg["mean"] - agg["std"],
                        agg["mean"] + agg["std"],
                        color=color, alpha=0.15)

    ax.set_title(lbl)
    ax.set_xlabel("Lookback")
    ax.grid(alpha=0.3)
    ax.set_axisbelow(True)
    if m in bounded_01:
        ax.set_ylim(0, 1.05)

axes[0].set_ylabel("Score")
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=len(company_order),
           frameon=False, bbox_to_anchor=(0.5, 1.02))
fig.suptitle(f"{MODEL_NAME} — Metric distributions across runs", y=1.06, fontsize=13)

plt.tight_layout()
plt.savefig(f"{SAVE_DIR}/{MODEL_NAME}_metric_distributions.png", dpi=150, bbox_inches="tight")
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




