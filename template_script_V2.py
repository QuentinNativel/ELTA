"""
    Python script to submit as a part of the project of ELTP 2020 course.
    
    This script serves as a template. Please use proper comments and meaningful variable names.
"""
import numpy as np
from sklearn import tree

from sklearn.ensemble import (AdaBoostClassifier, BaggingClassifier,
                              GradientBoostingClassifier,
                              RandomForestClassifier)
from sklearn.metrics import accuracy_score, f1_score

from dataset_tool import load_data, smote_training



"""
    Group Members:
        (1) Marine Sobas
        (2) Sunjidmaa Shagdarsuren
        (3) Quentin Nativel  
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


def model_decision_tree(X_train, y_train, X_test, y_test):
    X_smote, y_smote = smote_training(X_train, y_train)
    clf = tree.DecisionTreeClassifier()
    clf.fit(X_smote, y_smote)
    y_pred = clf.predict(X_test)
    dt_f1 = f1_score(y_test, y_pred, average="weighted")
    dt_accuracy = accuracy_score(y_test, y_pred)
    return dt_accuracy, dt_f1


def model_bagging(X_train, y_train, X_test, y_test):
    clf_bag = BaggingClassifier(tree.DecisionTreeClassifier(), max_samples=0.3)
    X_smote, y_smote = smote_training(X_train, y_train)
    clf_bag.fit(X_smote, y_smote)
    y_pred_bag = clf_bag.predict(X_test)
    bag_f1 = f1_score(y_test, y_pred_bag, average="weighted")
    bag_accuracy = accuracy_score(y_test, y_pred_bag)
    return bag_accuracy, bag_f1


def model_random_forest(X_train, y_train, X_test, y_test):
    X_smote, y_smote = smote_training(X_train, y_train)
    rf = RandomForestClassifier(
        n_estimators=160, min_samples_split=10, max_features=2, verbose=True, n_jobs=-1)
    rf.fit(X_smote, y_smote)
    y_pred_rf = rf.predict(X_test)
    rf_accuracy = accuracy_score(y_test, y_pred_rf)
    rf_f1 = f1_score(y_test, y_pred_rf, average="weighted")
    return rf_accuracy, rf_f1


def model_gradient_boosting(X_train, y_train, X_test, y_test):
    X_smote, y_smote = smote_training(X_train, y_train)
    gbc = GradientBoostingClassifier(n_estimators=162, learning_rate=0.08, min_samples_split=341, min_samples_leaf=31,
                                     max_depth=15, subsample=0.87, verbose=True)
    gbc.fit(X_smote, y_smote)
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

    dt_acc, dt_f1 = model_decision_tree(X_train, y_train, X_test, y_test)

    bag_acc, bag_f1 = model_bagging(X_train, y_train, X_test, y_test)

    rf_acc, rf_f1 = model_random_forest(X_train, y_train, X_test, y_test)

    gbc_acc, gbc_f1 = model_gradient_boosting(X_train, y_train, X_test, y_test)

    ada_acc, ada_f1 = model_adaboost(
        X_train, y_train, X_test, y_test)

    # print the results
    print("Decision tree", dt_acc, dt_f1)
    print("Bagging decision tree", bag_acc, bag_f1)
    print("Random forest", rf_acc, rf_f1)
    print("Gradient boosting tree", gbc_acc, gbc_f1)
    print("Adaboost", ada_acc, ada_f1)
