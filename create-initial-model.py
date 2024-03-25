import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam


# AIDAN DEGOOYER - JAN 28, 2024
#
# Works Cited:
# DATASET: https://www.kaggle.com/datasets/teejmahal20/airline-passenger-satisfaction
#
# https://keras.io/guides/training_with_built_in_methods/
# https://pandas.pydata.org/docs/reference/frame.html#dataframe
# https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.fillna.html
# https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.iloc.html
# https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html
# https://towardsdatascience.com/what-and-why-behind-fit-transform-vs-transform-in-scikit-learn-78f915cf96fe
# https://matplotlib.org/3.5.3/api/_as_gen/matplotlib.pyplot.html
# https://www.javatpoint.com/how-to-add-two-lists-in-python
# https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html
#
# Debugging: https://datascience.stackexchange.com/questions/67047/loss-being-outputed-as-nan-in-keras-rnn
#            https://stackoverflow.com/questions/49135929/keras-binary-classification-sigmoid-activation-function
#

# calculates the value of q for an epoch
def calculate_q(confusion_matrix, b1, b2):
    # Extract false positives (FP) and false negatives (FN)
    FP = confusion_matrix[0, 1]
    FN = confusion_matrix[1, 0]

    # Calculate q value
    q = (FP + FN) / (b1 + b2) + 10 * (max(0, FP / b1 - 0.1) + max(0, FN / b2 - 0.1))

    return q


# Load data from csv
train_data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")

# Remove id and number
train_data = train_data.iloc[:, 2:]
test_data = test_data.iloc[:, 2:]

# Input missing values with the mean
train_data.fillna(train_data.mean(), inplace=True)
test_data.fillna(test_data.mean(), inplace=True)

# set number of data points from each class
b1 = 1000
b2 = 3000

class_0_train = train_data[train_data['label'] == 0]
class_1_train = train_data[train_data['label'] == 1]

# Sample b1 data points from one class and b2 from the other
train_class_0 = resample(class_0_train, n_samples=b1, random_state=42)
train_class_1 = resample(class_1_train, n_samples=b2, random_state=42)
train_data = pd.concat([train_class_0, train_class_1])

# Randomly relabel 5% of the training data from each class
num_samples_class_0_relabel = int(0.05 * len(train_class_0))
num_samples_class_1_relabel = int(0.05 * len(train_class_1))

selected_indices_class_0_relabel = np.random.choice(train_class_0.index, size=num_samples_class_0_relabel,
                                                    replace=False)
selected_indices_class_1_relabel = np.random.choice(train_class_1.index, size=num_samples_class_1_relabel,
                                                    replace=False)

train_data.loc[selected_indices_class_0_relabel, 'label'] = 1
train_data.loc[selected_indices_class_1_relabel, 'label'] = 0

# Concatenate the relabeled data with the original data
train_data = pd.concat([train_class_0, train_class_1])

# Separate testing data into two classes based on labels
class_0_test = test_data[test_data['label'] == 0]
class_1_test = test_data[test_data['label'] == 1]

# Sample 100 data points from each class for testing
test_class_0 = class_0_test.sample(n=100, random_state=42)
test_class_1 = class_1_test.sample(n=100, random_state=42)
test_data = pd.concat([test_class_0, test_class_1])

# Split features and labels for training and testing sets
X_train = train_data.drop(columns=['label']).values
y_train = train_data['label'].values
X_test = test_data.drop(columns=['label']).values
y_test = test_data['label'].values

# standardize
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

input_dim = X_train.shape[1]


# function to train the model
def train_model(X_train, y_train, X_test, y_test, b1, b2):
    input_dim = X_train.shape[1]
    model = Sequential()
    model.add(Dense(units=input_dim * 2, activation='sigmoid'))
    model.add(Dense(units=1, activation='sigmoid'))
    optimizer = Adam(clipvalue=0.5)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    # sets up an array to catch all the q values from each epoch
    q_train_per_epoch = []
    q_test_per_epoch = []

    for epoch in range(epochs):
        # Train model for one epoch
        history = model.fit(X_train, y_train, epochs=1, batch_size=32, verbose=0)

        # Calculate q values for training and test data at this epoch
        y_pred_train = np.round(model.predict(X_train)).astype(int)
        y_pred_test = np.round(model.predict(X_test)).astype(int)
        confusion_matrix_train = confusion_matrix(y_train, y_pred_train)
        confusion_matrix_test = confusion_matrix(y_test, y_pred_test)

        q_train = calculate_q(confusion_matrix_train, b1, b2)
        q_test = calculate_q(confusion_matrix_test, 100, 100)

        train_loss, train_accuracy = model.evaluate(X_train, y_train, verbose=0)
        test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)

        q_train_per_epoch.append(q_train)
        q_test_per_epoch.append(q_test)

        print(
            f"Epoch {epoch + 1}: Train Loss = {train_loss:.4f}, Train Accuracy = {train_accuracy:.4f}, Test Loss = {test_loss:.4f}, Test Accuracy = {test_accuracy:.4f}")

    return history, q_train_per_epoch, q_test_per_epoch, model


# Repeat experiment
epochs = 50
num_experiments = 1

q_train_avg = np.zeros(epochs)
q_test_avg = np.zeros(epochs)
confusion_matrices_train = []
confusion_matrices_test = []

for i in range(num_experiments):
    history, q_train_per_epoch, q_test_per_epoch, trained_model = train_model(X_train, y_train, X_test, y_test, b1, b2)

    trained_model.save("model.keras")

    # add the q values from each epoch to a total from all experiments (will be used for averaging later)
    q_train_avg = [a + b for a, b in zip(q_train_avg, q_train_per_epoch)]
    q_test_avg = [a + b for a, b in zip(q_test_avg, q_test_per_epoch)]

    print(
        f"**********************************************EXPERIMENT NUMBER {i + 1} COMPLETE**********************************************")

    # Calculate confusion matrices
    y_pred_train = np.round(trained_model.predict(X_train)).astype(int)
    y_pred_test = np.round(trained_model.predict(X_test)).astype(int)
    confusion_matrices_train.append(confusion_matrix(y_train, y_pred_train))
    confusion_matrices_test.append(confusion_matrix(y_test, y_pred_test))

# Display confusion matrices
for i in range(num_experiments):
    print(f"Confusion Matrix - Training Data (Experiment {i + 1}):")
    print(confusion_matrices_train[i])
    print(f"Confusion Matrix - Test Data (Experiment {i + 1}):")
    print(confusion_matrices_test[i])

# averaging q values per epoch
q_train_final = []
for x in q_train_avg:
    q_train_final.append(x / num_experiments)
q_test_final = []
for x in q_test_avg:
    q_test_final.append(x / num_experiments)

# plot the graph using matplotlib

plt.figure(figsize=(10, 6))
plt.plot(np.log(range(1, epochs + 1)), q_train_final, label='Training Data')
plt.plot(np.log(range(1, epochs + 1)), q_test_final, label='Test Data')
plt.xlabel('log(Number of Weight Updates)')
plt.ylabel('Average q')
plt.title('Average q vs. log(Number of Weight Updates)')
plt.legend()
plt.show()