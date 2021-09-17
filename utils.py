from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_validate
import logging

def calcula_cross_val_scores(clf, X_trein, labels_trein, cv):
    scoring = ['balanced_accuracy','precision_macro', 'recall_macro', 'roc_auc', 'neg_mean_squared_error']
    mul_scores = cross_validate(clf, X_trein, labels_trein, scoring=scoring, cv=cv, return_train_score=True)
    acuracias_treino = mul_scores['train_balanced_accuracy']
    result = "train_balanced_accuracy:      {:.3f} (+/- {:.3f})".format(acuracias_treino.mean(), acuracias_treino.std())
    print (result)
    logging.info (result)
    acuracias_teste  = mul_scores['test_balanced_accuracy']
    result = "test_balanced_accuracy:       {:.3f} (+/- {:.3f})".format(acuracias_teste.mean(), acuracias_teste.std())
    print (result)
    logging.info (result)
    scores = mul_scores['test_precision_macro']
    result = "test_precision_macro:         {:.3f} (+/- {:.3f})".format(scores.mean(), scores.std())
    print (result)
    logging.info (result)
    scores = mul_scores['test_recall_macro']
    result = "test_recall_macro:            {:.3f} (+/- {:.3f})".format(scores.mean(), scores.std())
    print (result)
    logging.info (result)
    scores = mul_scores['test_roc_auc']
    result = "test_roc_auc:                 {:.3f} (+/- {:.3f})".format(scores.mean(), scores.std())
    print (result)
    logging.info (result)
    scores = mul_scores['train_neg_mean_squared_error']
    result = "train_neg_mean_squared_error: {:.3f} (+/- {:.3f})".format(-scores.mean(), scores.std())
    print (result)
    logging.info (result)
    scores = mul_scores['test_neg_mean_squared_error']
    result = "test_neg_mean_squared_error : {:.3f} (+/- {:.3f})".format(-scores.mean(), scores.std())
    print (result)
    logging.info (result)
    return acuracias_treino.mean(), acuracias_teste.mean()