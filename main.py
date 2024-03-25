import json

import keras

from individual import Individual
import pandas as pd
import numpy as np
import time
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Dense

# HYPERPARAMETERS
importdata = True
randomrelabel = False
generations = 33
num_experiments = 10
population_size = 44
num_parents = 4
mutation_rate = (2 / 1012)

start_time = time.time()
# shouldn't be used other than for testing
# input_dim = 20


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


# data parsing begins here======================================================================================
# Load data from csv
if importdata:
    train_data = pd.read_csv("train.csv")
    test_data = pd.read_csv("test.csv")

    # Remove id and number
    train_data = train_data.iloc[:, 2:]
    test_data = test_data.iloc[:, 2:]

    # Input missing values with the mean
    train_data.fillna(train_data.mean(), inplace=True)
    test_data.fillna(test_data.mean(), inplace=True)

    # set number of data points from each class
    b1 = 100
    b2 = 300

    class_0_train = train_data[train_data['label'] == 0]
    class_1_train = train_data[train_data['label'] == 1]


    def resample_data():
        global train_class_0
        global train_class_1
        global train_data
        train_class_0 = resample(class_0_train, n_samples=b1, random_state=42)
        train_class_1 = resample(class_1_train, n_samples=b2, random_state=42)
        train_data = pd.concat([train_class_0, train_class_1])
        global X_train
        global y_train
        X_train = train_data.drop(columns=['label']).values
        y_train = train_data['label'].values


    resample_data()

    # Sample b1 data points from one class and b2 from the other
    if randomrelabel:
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
    X_test = test_data.drop(columns=['label']).values
    y_test = test_data['label'].values

    # standardize
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    input_dim = X_train.shape[1]


# END data parsing ================================================================================================


# initialize population. Instantiates pop_size number of individuals, and calculates q from the individual.

def initialize_population(pop_size: int, input_dim):
    pop_list = []
    for i in range(pop_size):
        pop_list.append(Individual(input_dim))
        pop_list[i].randomize_vector()
        pop_list[i].mask_weights()
        pop_list[i].calculate_q(X_train, y_train, b1, b2)
        print(f"Individual #{i + 1} created.")
    pop_list.sort(key=lambda a: a.q)  # sorts population list by q value
    return pop_list


# evolve the population, returns a new generation sorted by q
def evolve_population(population, mutation_rate):
    offspring = []
    stud = population[0].get_vector()
    for j in range(population_size):
        parent2 = population[np.random.randint(0, high=(population_size - 1))].get_vector()
        offspring.append(Individual(input_dim))
        temp_vector = population[0].get_vector()
        for i in range(len(stud)):
            if np.random.rand() < 0.5:
                temp_vector[i] = parent2[i]

        offspring[j].set_vector(temp_vector)
        offspring[j].mutate(mutation_rate)
        offspring[j].repair()
        offspring[j].mask_weights()
        offspring[j].calculate_q(X_train, y_train, b1, b2)

    offspring.sort(key=lambda a: a.q)
    return offspring


def evaluate_model_conf_matrix(x_test, y_test, individual):
    # not really useful except when needing confusion matricies for a report
    current_model = individual.get_model()
    predictions = current_model.predict(x_test)
    rounded_predictions = np.rint(predictions)
    conf_matrix = confusion_matrix(y_test, rounded_predictions)  # needs help
    return conf_matrix


def genetic_algorithm(X_train, y_train, mutation_rate, num_generations):
    best_q_values_trial = []
    best_q_values_generations = {1: [], 2: [], 4: [], 8: [], 12: [], 16: [], 24: [], 32: []}
    models_with_best_q = [0] * num_experiments
    last_best_q = 9
    test_q_vals_per_generation = [0] * num_generations
    testModel = Individual(input_dim)

    for experiment_num in range(num_experiments):
        population = initialize_population(population_size, input_dim)
        best_q_value_trial = 999999999

        for generation in range(num_generations):
            generational_q = 9999999
            offspring = evolve_population(population, mutation_rate)

            for individual in offspring:
                if individual.q < generational_q:
                    generational_q = individual.q
                    generational_model = individual

                if individual.q < best_q_value_trial:
                    best_q_value_trial = individual.q
                    print(f"\n****NEW BEST Q: {individual.q:.2f}****\n\n----------------------------------------------")
                    models_with_best_q[experiment_num] = individual

            if generation in [1, 2, 4, 8, 12, 16, 24, 32]:
                best_q_values_generations[generation].append(best_q_value_trial)

            population = offspring

            # Adapt mutation rate using the modified 1/5 rule

            # evaluate best model of generation
            testModel.set_weights(generational_model.get_weights())
            testModel.calculate_q(X_test, y_test, 100, 100)
            test_q_vals_per_generation[generation] += testModel.q

            print(f"\n\nLAST MUTATION EVAL BEST Q = {last_best_q:.2f}")
            print(f"THIS GENERATION BEST Q = {generational_q:.2f}")
            print(f"MUTATION RATE = {mutation_rate:.2f}")
            print(f"GENERATION NUMBER {generation}")
            print(f"**TEST Q VAL {testModel.q}")
            print(f"EXPERIMENT NUMBER {experiment_num + 1}\n\n----------------------------------------------")

        best_q_values_trial.append(best_q_value_trial)
        print(f"**EXPERIMENT NUMBER {experiment_num + 1} COMPLETE**")

    return best_q_values_trial, best_q_values_generations, models_with_best_q, test_q_vals_per_generation


best_qs_trial, q_selected_gens, models_with_best_q, test_q_vals = genetic_algorithm(X_train, y_train, mutation_rate,
                                                                                    generations)

print("--- %s seconds ---" % (time.time() - start_time))

file = open("output.txt", "w")

for i, individual in enumerate(models_with_best_q):
    file.write((f"Experiment #{i + 1} - Train"))
    conf_matrix = evaluate_model_conf_matrix(X_train, y_train, individual)
    file.write(f"\n{conf_matrix}\n")

    file.write((f"Experiment #{i + 1} - Test"))
    conf_matrix = evaluate_model_conf_matrix(X_test, y_test, individual)
    file.write(f"\n{conf_matrix}\n")

file.write("---------------------\nbest q's in trials\n")

for item in best_qs_trial:
    file.write(f"\n{item}\n")

file.write("---------------------\naverage test q's per generation\n")

outList = [item / num_experiments for item in test_q_vals]
for item in outList:
    file.write(str(item))
    file.write("\n")

file.write("---------------------\n")

unmasked_model = Individual(input_dim)
unmasked_model.calculate_q(X_train, y_train, b1, b2)
file.write(f"Baseline q value: {unmasked_model.q}\n")

file.write("---------------------\n")

file.write(json.dumps(q_selected_gens))
