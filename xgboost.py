import xgboost as xgb
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

raw_test_data = []
X_train = []
y_train = []
X_cv = []
y_cv = []
X_test = []


# Create a DMatrix for XGBoost
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_cv, label=y_cv)

# Define XGBoost parameters
params = {
    "objective": "binary:logistic",
    "eval_metric": "logloss",  # Use log loss as the evaluation metric
    "eta": 0.05,  # Learning rate
    "max_depth": 8,  # Maximum depth of the tree
    "min_child_weight": 1,  # Minimum sum of instance weight (hessian) needed in a child
    "subsample": 0.8,  # Subsample ratio of the training data
    "colsample_bytree": 0.8,  # Subsample ratio of columns when constructing each tree
    "seed": 42,
}

# Define a watchlist to monitor performance on the validation set
watchlist = [(dtrain, "train"), (dtest, "test")]

num_round = 500  # Number of boosting rounds
evals_result = {}  # Create an empty dictionary to store evaluation results

model = xgb.train(
    params,
    dtrain,
    num_round,
    evals=watchlist,
    early_stopping_rounds=10,
    verbose_eval=True,
    evals_result=evals_result  # Capture evaluation results
)

# To make predictions
y_pred = model.predict(dtest)

# You can adjust the classification threshold based on your requirements
threshold = 0.5
y_pred_binary = np.where(y_pred > threshold, 1, 0)

def plot_xgboost_training_history(evals_result, num_round):
    """
    Plot training history for XGBoost.

    Args:
        evals_result: The evaluation results containing training history.
        num_round: The number of boosting rounds.

    Returns:
        None
    """
    train_metric_name = "logloss"  # Use log loss as the evaluation metric
    val_metric_name = "logloss"  # Update this to match the actual metric name

    # Get the number of boosting rounds actually used
    num_actual_rounds = len(evals_result["train"][train_metric_name])

    # Ensure that num_round is within the valid range
    num_round = min(num_round, num_actual_rounds)

    # Create an array of epochs corresponding to the number of rounds
    epochs = range(1, num_round + 1)

    # Extract training and validation metrics
    train_metrics = evals_result["train"][train_metric_name][:num_round]
    val_metrics = evals_result["test"][val_metric_name][:num_round]

    # Calculate accuracy at each round
    train_accuracy = [1 - metric for metric in train_metrics]
    val_accuracy = [1 - metric for metric in val_metrics]

    # Plot training and validation metrics
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_metrics, 'b', label=f'Training {train_metric_name}')
    plt.plot(epochs, val_metrics, 'r', label=f'Validation {val_metric_name}')
    plt.title(f'Training and Validation {train_metric_name}')
    plt.xlabel('Boosting Rounds')
    plt.ylabel(train_metric_name)
    plt.legend()

    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracy, 'b', label='Training Accuracy')
    plt.plot(epochs, val_accuracy, 'r', label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Boosting Rounds')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.show()

def plot_xgboost_confusion_matrix(cm):
    """
    Display a confusion matrix plot for XGBoost.

    Args:
        cm: Confusion matrix obtained from predictions.

    Returns:
        None
    """
    display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.unique(y_cv))
    display.plot(cmap=plt.cm.Blues, values_format='d', xticks_rotation='vertical')
    plt.title('Confusion Matrix')
    plt.show()


# Make predictions on the validation set
y_cv_prediction = model.predict(dtest, ntree_limit=model.best_iteration)

# Convert probabilities to binary labels based on the threshold
threshold = 0.6
y_cv_prediction_rounded = np.where(y_cv_prediction > threshold, 1, 0)

# Calculate the confusion matrix using y_cv (validation labels) and y_cv_prediction_rounded
cm = confusion_matrix(y_cv, y_cv_prediction_rounded)

# Specify the number of rounds for the accuracy plot
num_round = 200  # Set this to the desired number of rounds

plot_xgboost_confusion_matrix(cm)
plot_xgboost_training_history(evals_result, num_round)


# Make predictions on the test set
dtest_final = xgb.DMatrix(X_test)
y_test_prediction = model.predict(dtest_final, ntree_limit=model.best_iteration)

# Convert probabilities to binary labels based on the threshold
threshold = 0.6
y_test_prediction_rounded = np.where(y_test_prediction > threshold, 1, 0)

# Create a DataFrame with test IDs and predicted labels
test_ids = raw_test_data.iloc[:, 0]  # Get only the IDs from the test dataset
result = pd.DataFrame({'PassengerId': test_ids, 'Survived': y_test_prediction_rounded})

# Save the result to a CSV file
result.to_csv('./solutions/xgboost_kaggle_titanic.csv', index=False)
