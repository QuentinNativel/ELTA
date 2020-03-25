from sklearn.metrics import f1_score, accuracy_score, make_scorer
from sklearn.model_selection import cross_val_score


def test_score(model, X_test, y_test, metric="weighted f1 score"):
    y_pred = model.predict(X_test)
    if metric == "weighted f1 score":
        return f1_score(y_test, y_pred, average="weighted")
    elif metric == "accuracy":
        return accuracy_score(y_test, y_pred)
    else:
        print("error : the metric's spelling is wrong, please write either 'weighted f1 score' or 'accuracy' ")


def cross_validation(model, X, y, scorer="weighted f1 score"):
    if scorer == "weighted f1 score":
        f1_scorer = make_scorer(f1_score, average="weighted")
        return cross_val_score(model, X, y, scoring=f1_scorer, cv=3)
    elif scorer == "loss":
        return cross_val_score(model, X, y, cv=3)
    else:
        print("error : the scorer's spelling is wrong, please write either 'weighted f1 score' or 'loss'")
