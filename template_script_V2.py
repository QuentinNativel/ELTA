"""
    Python script to submit as a part of the project of ELTP 2020 course.
    
    This script serves as a template. Please use proper comments and meaningful variable names.
"""
import numpy as np
from sklearn import tree
from sklearn.ensemble import (AdaBoostClassifier, GradientBoostingClassifier,
                              RandomForestClassifier)
from sklearn.metrics import accuracy_score, f1_score

from dataset_tool import load_data


"""
    Group Members:
        (1) ...
        (2) ...  
        etc.
"""

"""
    Import necessary packages
"""


"""
        etc.
"""

"""
    Your methods implementing the models.
    
    Each of your model should have a separate method. e.g. run_random_forest, run_decision_tree etc.
    
    Your method should:
        (1) create the proper instance of the model with the best hyperparameters you found
        (2) fit the model with a given training data
        (3) run the prediction on a given test data
        (4) return accuracy and F1 score
        
    Following is a sample method. Please note that the parameters given here are just examples.
"""


def model_random_forest(X_train, y_train, X_test, y_test):
    """
    @param: X_train - a numpy matrix containing features for training data (e.g. TF-IDF matrix)
    @param: y_train - a numpy array containing labels for each training sample
    @param: X_test - a numpy matrix containing features for test data (e.g. TF-IDF matrix)
    @param: y_test - a numpy array containing labels for each test sample
    """
    clf = RandomForestClassifier(n_estimators=100, max_depth=2,
                                 random_state=0)  # please choose all necessary parameters
    clf.fit(X_train, y_train)

    y_predicted = clf.predict(X_test)
    rf_accuracy = accuracy_score(y_test, y_predicted)
    rf_f1 = f1_score(y_test, y_predicted, average="weighted")

    return rf_accuracy, rf_f1


def model_gradient_boosting(X_train, y_train, X_test, y_test):
    gbc = GradientBoostingClassifier(n_estimators=162, learning_rate=0.08, min_samples_split=341, min_samples_leaf=31,
                                     max_depth=15, subsample=0.87, random_state=10, verbose=True)
    gbc.fit(X_train, y_train)

    y_predicted = gbc.predict(X_test)
    gbc_accuracy = accuracy_score(y_test, y_predicted)
    gbc_f1 = f1_score(y_test, y_predicted, average="weighted")

    return gbc_accuracy, gbc_f1


def model_adaboost(X_train, y_train, X_test, y_test):
    estimators = 100
    alpha = 0.1
    random = None
    ada = AdaBoostClassifier(base_estimator=tree.DecisionTreeClassifier(max_depth=300),
                             n_estimators=estimators, learning_rate=alpha, random_state=random)

    ada.fit(X_train, y_train)
    y_pred = ada.predict(X_test)
    ada_acc = accuracy_score(y_test, y_pred)
    ada_f1 = f1_score(y_test, y_pred, average="weighted")
    return ada_acc, ada_f1


"""
   The main function should print all the accuracies and F1 scores for all the models.
   
   The names of the models should be sklearn classnames, e.g. DecisionTreeClassifier, RandomForestClassifier etc.
   
   Please make sure that your code is outputting the performances in proper format, because your script will be run automatically by a meta-script.
"""
if __name__ == "__main__":
    """
       This is just an example, plese change as necceary. Just maintain final output format with proper names of the models as described above.
    """
    x_path = "X_train_update.csv"
    y_path = "Y_train_CVw08PX.csv"
    X_train, X_test, y_train, y_test = load_data(x_path, y_path, test=True)

    gbc_acc, gbc_f1 = model_gradient_boosting(X_train, y_train, X_test, y_test)
    ada_acc, ada_f1 = model_adaboost(
        X_train, y_train, X_test, y_test)
    #model_1_acc, model_1_f1 = run_model_1(...)
    #model_2_acc, model_2_f1 = run_model_2(...)
    """
        etc.
    """

    # print the results
    #print("model_1", model_1_acc, model_1_f1)
    #print("model_2", model_2_acc, model_2_f1)
    print("Gradient boosting tree", gbc_acc, gbc_f1)
    print("Adaboost", ada_acc, ada_f1)
    """
        etc.
    """
