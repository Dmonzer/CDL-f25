#!/usr/bin/env python
# coding: utf-8

# In[17]:


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
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
import os
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score
import itertools
from sklearn.metrics import accuracy_score, classification_report
import keras_tuner as kt
from tensorflow.keras.optimizers import Adam
import os
from sklearn.utils.class_weight import compute_class_weight

'''
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
'''
# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[18]:


#df= pd.read_csv("processed_news_scores_with_stocks_WITH_AGGREGATE_ready_formodeling.csv")
#score_cols = [f"scores_{i}" for i in range(23)]
#df = df.drop(columns=score_cols)
df_tesla = pd.read_csv("Tesla_processed_news_scores_with_stocks_WITH_AGGREGATE_ready_formodeling.csv")
df_apple= pd.read_csv("Apple_processed_news_scores_with_stocks_WITH_AGGREGATE_ready_formodeling.csv")
df_amazon= pd.read_csv("Amazon_processed_news_scores_with_stocks_WITH_AGGREGATE_ready_formodeling.csv")


# In[19]:


df_tesla = df_tesla.sort_values(by="timestamp").reset_index(drop=True)
df_apple = df_apple.sort_values(by="timestamp").reset_index(drop=True)
df_amazon = df_amazon.sort_values(by="timestamp").reset_index(drop=True)


# In[20]:


df_tesla.isna().any()


# In[21]:


df_tesla['close_sign'] = (df_tesla['close'] > df_tesla['close'].shift(1)).astype(int)
df_tesla['close_diff'] = (df_tesla['close'].diff()).astype(float)
df_tesla['close_diff'] = df_tesla['close_diff'].fillna(0)


df_apple['close_sign'] = (df_apple['close'] > df_apple['close'].shift(1)).astype(int)
df_apple['close_diff'] = (df_apple['close'].diff()).astype(float)
df_apple['close_diff'] = df_apple['close_diff'].fillna(0)



df_amazon['close_sign'] = (df_amazon['close'] > df_amazon['close'].shift(1)).astype(int)
df_amazon['close_diff'] = (df_amazon['close'].diff()).astype(float)
df_amazon['close_diff'] = df_amazon['close_diff'].fillna(0)


# In[22]:


df_amazon['close_diff'].hist()#.value_counts(bins=10).sort_index()


# In[23]:


df_amazon['close_diff'].quantile([0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])


# In[24]:


amazon_thresholds = [-np.inf, -5, -2, 0, 2, 5, np.inf]
amazon_labels = ["-3", "-2", "-1", "1", "2", "3"]

df_amazon["diff_magnitude"] = pd.cut(
    df_amazon["close_diff"],
    bins=amazon_thresholds,
    labels=amazon_labels
)


# In[25]:


df_apple['close_diff'].quantile([0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])


# In[26]:


apple_thresholds = [-np.inf, -1, -0.25,0, 0.25, 1, np.inf]
apple_labels = ["-3", "-2", "-1", "1", "2", "3"]

df_apple["diff_magnitude"] = pd.cut(
    df_apple["close_diff"],
    bins=apple_thresholds,
    labels=apple_labels
)


# In[27]:


df_tesla['close_diff'].quantile([0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])


# In[28]:


tesla_thresholds = [-np.inf, -5, -1,0, 1, 5, np.inf]
tesla_labels = ["-3", "-2", "-1", "1", "2", "3"]

df_tesla["diff_magnitude"] = pd.cut(
    df_tesla["close_diff"],
    bins=tesla_thresholds,
    labels=tesla_labels
)


# In[29]:


df_tesla['close_diff'].hist(bins=30)


# In[30]:


df_amazon['close'].value_counts(bins=10).sort_index()


# In[31]:


# Map label strings -> integer indices
label_map = {"-3": 0, "-2": 1, "-1": 2, "1": 3, "2": 4, "3": 5}
for df_x in [df_tesla, df_apple, df_amazon]:
    df_x["diff_magnitude_idx"] = df_x["diff_magnitude"].map(label_map).astype(int)


# In[ ]:





# In[ ]:





# # Transformer_Encoder

# ## Close price with avg min, and max score of all. 

# #### Changes implemented on nov 30 code:
# 1- Adding stopping epochs. 
# 
# 2- add validation dataset
# 
# 
# 3- NO NORMALIZATION to target . ALL ARE WITHIN 0 AND 1. only normalize features. 
# 
# 4- Adding evaluation metrics 
# 
# 5- close diff is added to features. 
# 
# Limitations to note:
# - the time series, does not have same interval gaps due to market closure, for example some datapoints might have 1 hour gaps while others have 9. News are accumulated within the 9 hours of market closure and treated equally. otherwise the datapoint gaps are treated similarly. 

# ## hYPERTUNING 
# 
# Hypertuning Strategy Summary
# Approach
# 
# - Library: Keras Tuner (keras_tuner)
# - Search algorithm: Bayesian Optimization (smarter than random search — learns from previous trials)
# - Tuning frequency: Once per company (3 total tuning runs per model version), then re-used across all lookback × train_size × repeat combinations
# 
# Hyperparameters Tuned
# Hyperparameter  Search Space
# - Number of LSTM layers1 or 2 (stacked)
# - LSTM units25, 50, 100
# - Dropout rate0.1, 0.2, 0.3
# - Learning rate0.001, 0.01
# 
# Fixed During Tuning
# 
# - Lookback: 30 (sensible default to keep tuning fast)
# - Train size: 0.7 (standard split)
#  - Batch size: 240 (kept fixed)
# - Tuning epochs: 50 (with early stopping)
# - Max trials: 15 per company
# 
# Workflow
# 
# - Split data using fixed lookback=30 and ts=0.7
# - Normalize after split (fit on train only)
# - Run Bayesian Optimization for up to 15 trials, optimizing val_loss (regression) or val_accuracy (classification)
# - Store best hyperparameters per company in a dict
# - Use those hyperparameters in the main n_repeats × lookbacks × train_sizes loop
# 
# ## MULTICLASS:
# - ADDING CLASS WEIGHTS FOR BALANCING. 
# - 6 CLASSES BASED ON PERCENTILES. 

# In[ ]:


from tensorflow.keras.layers import (Input, MultiHeadAttention, LayerNormalization,
                                     Dense, Dropout, GlobalAveragePooling1D)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import keras_tuner as kt
import os
os.makedirs("nov30_codes_new_data/TRANSFORMER-HYPERTUNING", exist_ok=True)

companies = {"Tesla": df_tesla, "Apple": df_apple, "Amazon": df_amazon}
n_classes = 6
label_names = ["-3", "-2", "-1", "1", "2", "3"]


def create_dataset_xy(X, y, look_back):
    Xs, Ys = [], []
    for i in range(len(X) - look_back):
        Xs.append(X[i:i+look_back])
        Ys.append(y[i+look_back])
    return np.array(Xs), np.array(Ys)

early_stop = EarlyStopping(monitor="val_loss", patience=20, restore_best_weights=True)


# ===== HYPERTUNING SECTION (Transformer univariate) =====
TUNE_LOOKBACK = 30
TUNE_TS = 0.7

best_hyperparams_per_company = {}

for company, df in companies.items():
    print(f"\n=== Hypertuning {company} (Transformer univariate) ===")

    X_features = df[['close_diff']].values
    y_target   = df['diff_magnitude_idx'].values.astype(int)

    train_size = int(len(X_features) * TUNE_TS)
    X_train, X_test = X_features[:train_size], X_features[train_size:]
    y_train, y_test = y_target[:train_size],   y_target[train_size:]

    scaler = MinMaxScaler(feature_range=(0, 1))
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    trainX, trainY = create_dataset_xy(X_train, y_train, TUNE_LOOKBACK)
    n_features = trainX.shape[2]

    class_weights = compute_class_weight("balanced", classes=np.unique(trainY), y=trainY)
    class_weight_dict = dict(enumerate(class_weights))

    def build_model(hp):
        head_size = hp.Choice("head_size", [32, 64, 128])
        num_heads = hp.Choice("num_heads", [2, 4])
        ff_dim    = hp.Choice("ff_dim", [32, 64, 128])
        n_blocks  = hp.Int("n_blocks", 1, 2)
        dropout   = hp.Choice("dropout", [0.1, 0.2, 0.3])
        lr        = hp.Choice("lr", [1e-3, 1e-2])

        inputs = Input(shape=(TUNE_LOOKBACK, n_features))
        x = inputs
        for _ in range(n_blocks):
            attn = MultiHeadAttention(num_heads=num_heads, key_dim=head_size, dropout=dropout)(x, x)
            x = LayerNormalization(epsilon=1e-6)(attn + x)
            ff = Dense(ff_dim, activation="relu")(x)
            ff = Dropout(dropout)(ff)
            ff = Dense(n_features)(ff)
            x = LayerNormalization(epsilon=1e-6)(ff + x)
        x = GlobalAveragePooling1D()(x)
        x = Dropout(dropout)(x)
        outputs = Dense(n_classes, activation="softmax")(x)

        model = Model(inputs, outputs)
        model.compile(loss="sparse_categorical_crossentropy",
                      optimizer=Adam(learning_rate=lr),
                      metrics=["accuracy"])
        return model

    tuner = kt.BayesianOptimization(
        build_model,
        objective="val_accuracy",
        max_trials=15,
        directory="nov30_codes_new_data/TRANSFORMER-HYPERTUNING",
        project_name=f"transformer_univariate_{company}",
        overwrite=True,
    )

    tuner.search(
        trainX, trainY,
        epochs=50,
        validation_split=0.1,
        batch_size=240,
        callbacks=[early_stop],
        class_weight=class_weight_dict,
        verbose=0,
    )

    best_hp = tuner.get_best_hyperparameters(1)[0]
    best_hyperparams_per_company[company] = {
        "head_size": best_hp.get("head_size"),
        "num_heads": best_hp.get("num_heads"),
        "ff_dim":    best_hp.get("ff_dim"),
        "n_blocks":  best_hp.get("n_blocks"),
        "dropout":   best_hp.get("dropout"),
        "lr":        best_hp.get("lr"),
    }
    print(f"Best for {company}: {best_hyperparams_per_company[company]}")
    
    
n_repeats = 3
lookbacks = np.arange(15, 100, 15)
train_sizes = [0.7]#[0.6, 0.7, 0.8]
rows = []

for company, df in companies.items():
    best_acc = -float("inf")
    worst_acc = float("inf")

    X_features = df[['close_diff']].values   # univariate (use multivariate cols here for that version)
    y_target   = df['diff_magnitude_idx'].values.astype(int)

    for lookback, ts in itertools.product(lookbacks, train_sizes):
        print(company, lookback, ts)
        for repeat in range(n_repeats):

            np.random.seed(5)

            train_size = int(len(X_features) * ts)
            X_train, X_test = X_features[:train_size], X_features[train_size:]
            y_train, y_test = y_target[:train_size],   y_target[train_size:]

            scaler = MinMaxScaler(feature_range=(0, 1))
            X_train = scaler.fit_transform(X_train)
            X_test  = scaler.transform(X_test)

            trainX, trainY = create_dataset_xy(X_train, y_train, lookback)
            testX,  testY  = create_dataset_xy(X_test,  y_test,  lookback)
            n_features = trainX.shape[2]

            class_weights = compute_class_weight("balanced", classes=np.unique(trainY), y=trainY)
            class_weight_dict = dict(enumerate(class_weights))

            hp = best_hyperparams_per_company[company]

            inputs = Input(shape=(lookback, n_features))
            x = inputs
            for _ in range(hp["n_blocks"]):
                attn = MultiHeadAttention(num_heads=hp["num_heads"], key_dim=hp["head_size"], dropout=hp["dropout"])(x, x)
                x = LayerNormalization(epsilon=1e-6)(attn + x)
                ff = Dense(hp["ff_dim"], activation="relu")(x)
                ff = Dropout(hp["dropout"])(ff)
                ff = Dense(n_features)(ff)
                x = LayerNormalization(epsilon=1e-6)(ff + x)
            x = GlobalAveragePooling1D()(x)
            x = Dropout(hp["dropout"])(x)
            outputs = Dense(n_classes, activation="softmax")(x)

            model = Model(inputs, outputs)
            model.compile(loss="sparse_categorical_crossentropy",
                          optimizer=Adam(learning_rate=hp["lr"]),
                          metrics=["accuracy"])

            model.fit(trainX, trainY,
                      epochs=1000,
                      batch_size=240,
                      verbose=0,
                      validation_split=0.1,
                      callbacks=[early_stop],
                      class_weight=class_weight_dict)

            trainPredict = model.predict(trainX)
            testPredict  = model.predict(testX)

            trainPredClass = np.argmax(trainPredict, axis=1)
            testPredClass  = np.argmax(testPredict, axis=1)
            trainY_class   = trainY.astype(int).reshape(-1)
            testY_class    = testY.astype(int).reshape(-1)

            y_true = np.concatenate([trainY_class, testY_class])
            y_pred = np.concatenate([trainPredClass, testPredClass])

            accuracy  = accuracy_score(y_true, y_pred)
            precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
            recall    = recall_score(y_true, y_pred, average="macro", zero_division=0)
            f1        = f1_score(y_true, y_pred, average="macro", zero_division=0)
            try:
                from sklearn.preprocessing import label_binarize
                y_true_bin = label_binarize(y_true, classes=list(range(n_classes)))
                y_score = np.concatenate([trainPredict, testPredict], axis=0)
                auc = roc_auc_score(y_true_bin, y_score, average="macro", multi_class="ovr")
            except ValueError:
                auc = np.nan

            if accuracy > best_acc:
                best_acc = accuracy
                '''
                cm = confusion_matrix(y_true, y_pred, labels=list(range(n_classes)))
                disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["-3", "-2", "-1", "1", "2", "3"])
                disp.plot(cmap=plt.cm.Greens)
                plt.title(f"Transformer Confusion Matrix - {company}")
                plt.savefig(f"nov30_codes_new_data/TRANSFORMER-HYPERTUNING/Transformer_best_acc_{company}_lb{lookback}_ts{ts}_repeat{repeat}.png")

                cm_percent = cm.astype('float') / cm.sum() * 100
                disp = ConfusionMatrixDisplay(confusion_matrix=cm_percent, display_labels=["-3", "-2", "-1", "1", "2", "3"])
                disp.plot(cmap=plt.cm.Greens, values_format=".2f")
                plt.title(f"Transformer Confusion Matrix (%) - {company}")
                plt.savefig(f"nov30_codes_new_data/TRANSFORMER-HYPERTUNING/Transformer_best_acc_pct_{company}_lb{lookback}_ts{ts}_repeat{repeat}.png")
                '''
            if accuracy < worst_acc:
                worst_acc = accuracy
                '''
                cm = confusion_matrix(y_true, y_pred, labels=list(range(n_classes)))
                disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["-3", "-2", "-1", "1", "2", "3"])
                disp.plot(cmap=plt.cm.Greens)
                plt.title(f"Transformer Confusion Matrix - {company}")
                plt.savefig(f"nov30_codes_new_data/TRANSFORMER-HYPERTUNING/Transformer_worst_acc_{company}_lb{lookback}_ts{ts}_repeat{repeat}.png")
                '''
            result = {
                "company": company,
                "n_repeat": repeat,
                "lookback": lookback,
                "test_size": 1-ts,
                "test_acc": accuracy,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "auc": auc,
            }
            rows.append(result)

df_results = pd.DataFrame(rows)


# In[ ]:





# In[ ]:





# In[ ]:


df_results.to_csv("nov30_codes_new_data/Transformer_Encoder/Transformer_Encoder_only_MULTICLASS_Classification_v2.csv", index=False)


# In[ ]:


import matplotlib.pyplot as plt
import numpy as np

# ===== Summary plot: metrics per company =====
MODEL_NAME = "Transformer Encoder"   # change per model so files don't overwrite
SAVE_DIR   = "nov30_codes_new_data/Transformer_Encoder"

metrics = ["test_acc", "precision", "recall", "f1", "auc"]
metric_labels = ["Accuracy", "Precision", "Recall", "F1", "AUC"]

summary_mean = df_results.groupby("company")[metrics].mean()
summary_std  = df_results.groupby("company")[metrics].std()

# Keep a stable company order
company_order = ["Tesla", "Apple", "Amazon"]
summary_mean = summary_mean.reindex(company_order)
summary_std  = summary_std.reindex(company_order)

x = np.arange(len(company_order))
width = 0.15

fig, ax = plt.subplots(figsize=(10, 6))
colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(metrics)))

for i, (m, lbl) in enumerate(zip(metrics, metric_labels)):
    offset = (i - (len(metrics) - 1) / 2) * width
    bars = ax.bar(
        x + offset,
        summary_mean[m].values,
        width,
        yerr=summary_std[m].values,
        capsize=3,
        label=lbl,
        color=colors[i],
        edgecolor="black",
        linewidth=0.5,
    )
    # value labels on top of each bar
    for b, v in zip(bars, summary_mean[m].values):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.01,
                f"{v:.2f}", ha="center", va="bottom", fontsize=8)

ax.set_xticks(x)
ax.set_xticklabels(company_order)
ax.set_ylabel("Score")
ax.set_ylim(0, 1.05)
ax.set_title(f"{MODEL_NAME} — Metrics per Company (mean ± std over runs)")
ax.legend(loc="upper right", ncol=len(metrics), fontsize=9, frameon=False)
ax.grid(axis="y", alpha=0.3)
ax.set_axisbelow(True)

plt.tight_layout()
plt.savefig(f"{SAVE_DIR}/{MODEL_NAME}_summary_metrics_per_company.png", dpi=150)
plt.show()

# Optional: also save the aggregated table for later cross-model comparison
summary_table = summary_mean.copy()
summary_table.columns = [f"{c}_mean" for c in summary_table.columns]
for c in metrics:
    summary_table[f"{c}_std"] = summary_std[c]
summary_table.to_csv(f"{SAVE_DIR}/{MODEL_NAME}_summary_metrics_per_company.csv")


# In[ ]:


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

        # aggregate across repeats at each lookback
        agg = sub.groupby("lookback")[m].agg(["mean", "std", "count"]).reset_index()
        agg["std"] = agg["std"].fillna(0)

        color = company_colors[company]

        # individual run points (jittered slightly so overlapping points are visible)
        jitter = (np.random.RandomState(0).rand(len(sub)) - 0.5) * 1.5
        ax.scatter(sub["lookback"].values + jitter, sub[m].values,
                   color=color, alpha=0.35, s=25, edgecolor="none")

        # mean line + std shading
        ax.plot(agg["lookback"], agg["mean"], color=color, lw=2, label=company, marker="o")
        ax.fill_between(agg["lookback"],
                        agg["mean"] - agg["std"],
                        agg["mean"] + agg["std"],
                        color=color, alpha=0.15)

    ax.set_title(lbl)
    ax.set_xlabel("Lookback")
    ax.grid(alpha=0.3)
    ax.set_axisbelow(True)
    if m in ("test_acc", "precision", "recall", "f1", "auc"):
        ax.set_ylim(0, 1.05)

axes[0].set_ylabel("Score")
# single legend for the whole figure
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




