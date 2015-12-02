'''
Created on Jun 11, 2013

@author: navin.kolambkar
'''
from __future__ import division
from collections import defaultdict
from sklearn.cross_validation import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
import numpy as np
import data_io

if __name__ == '__main__':
    print("Getting features for deleted papers from the database")
    features_deleted = data_io.get_features_db("TrainDeleted")

    print("Getting features for confirmed papers from the database")
    features_conf = data_io.get_features_db("TrainConfirmed")

    features = [x[2:] for x in features_deleted + features_conf]
    target = [0 for x in range(len(features_deleted))] + [1 for x in range(len(features_conf))]
    
    X = np.array(features)
    y = np.array(target)
    
    print("Getting features for valid papers from the database")
    data = data_io.get_features_db("ValidPaper")
    author_paper_ids = [x[:2] for x in data]
    featuresTest = [x[2:] for x in data]
    
    X_submission = np.array(featuresTest)
    
    np.random.seed(0) # seed to shuffle the train set

    n_folds = 10
    
    skf = list(StratifiedKFold(y, n_folds))

    #===========================================================================
    # clfs = [RandomForestClassifier(n_estimators=100, n_jobs=-1, criterion='gini'),
    #        RandomForestClassifier(n_estimators=100, n_jobs=-1, criterion='entropy'),
    #        ExtraTreesClassifier(n_estimators=100, n_jobs=-1, criterion='gini'),
    #        ExtraTreesClassifier(n_estimators=100, n_jobs=-1, criterion='entropy'),
    #        GradientBoostingClassifier(learn_rate=0.05, subsample=0.5, max_depth=6, n_estimators=50)]
    #===========================================================================

    clfs = [RandomForestClassifier(n_estimators=100),
            GradientBoostingClassifier(learn_rate=0.05, subsample=0.5, max_depth=6, n_estimators=50),
            ExtraTreesClassifier(n_estimators=100)]
    
    print "Creating train and test sets for blending."
    
    dataset_blend_train = np.zeros((X.shape[0], len(clfs)))
    dataset_blend_test = np.zeros((X_submission.shape[0], len(clfs)))
    
    for j, clf in enumerate(clfs):
        print j, clf
        dataset_blend_test_j = np.zeros((X_submission.shape[0], len(skf)))
        for i, (train, test) in enumerate(skf):
            print "Fold", i
            X_train = X[train]
            y_train = y[train]
            X_test = X[test]
            y_test = y[test]
            clf.fit(X_train, y_train)
            y_submission = clf.predict_proba(X_test)[:,1]
            dataset_blend_train[test, j] = y_submission
            dataset_blend_test_j[:, i] = clf.predict_proba(X_submission)[:,1]
        dataset_blend_test[:,j] = dataset_blend_test_j.mean(1)
    
    print "Blending."
    clf = LogisticRegression()
    clf.fit(dataset_blend_train, y)
    y_submission = clf.predict_proba(dataset_blend_test)[:,1]

    print "Linear stretch of predictions to [0,1]"
    y_submission = (y_submission - y_submission.min()) / (y_submission.max() - y_submission.min())
    
    predictions = list(y_submission)
    author_predictions = defaultdict(list)
    paper_predictions = {}

    for (a_id, p_id), pred in zip(author_paper_ids, predictions):
        author_predictions[a_id].append((pred, p_id))

    for author_id in sorted(author_predictions):
        paper_ids_sorted = sorted(author_predictions[author_id], reverse=True)
        paper_predictions[author_id] = [x[1] for x in paper_ids_sorted]

    print("Writing predictions to file")
    data_io.write_submission(paper_predictions)