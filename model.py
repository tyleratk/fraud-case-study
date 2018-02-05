import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.linear_model import LogisticRegression
import pickle
import clean_data as cd

class ModelSelector():
    def __init__(self, models, params):
        self.models = models
        self.params = params
        self.model_names = models.keys()
        self.grid_searches = {}
        self.scores = {}

    def fit(self, X, y, verbose=1, scoring='f1', cv=3, n_jobs=-1):
        for name in self.model_names:
            print("Running GridSearchCV for {0}.".format(name))
            model = self.models[name]
            p = self.params[name]
            gs = GridSearchCV(model, p, cv=cv, n_jobs=n_jobs,
                              verbose=verbose, scoring=scoring)
            gs.fit(X,y)
            self.grid_searches[name] = gs
            self.scores[name] = gs.best_score_

    def score_summary(self):
        for name in self.model_names:
            print('{0} : F1 score = {1}\n'.format(name, self.scores[name]))


if __name__ == '__main__':

    df = pd.read_json('data/data.json')
    clean_df = cd.clean_data(df, True)

    y = clean_df.pop('fraud')
    X = clean_df.values
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y)

    X_train_new, y_train_new = cd.get_training_data(X_train, y_train)


    models = {'Random Forest': RandomForestClassifier(),
        'AdaBoost': AdaBoostClassifier(),
        'Gradient Boosting': GradientBoostingClassifier(),
        # 'SVC Linear': SVC(kernel='linear'),
        # 'SVC RBF' : SVC(kernel='rbf'),
        # 'Logistic Regression' : LogisticRegression(),
        'kNN' : KNeighborsClassifier()
    }

    params = {
        'Random Forest': {'n_estimators':[100]},
        'AdaBoost':  {},
        'Gradient Boosting': {},
        # 'SVC Linear': {},
        # 'SVC RBF' : {},
        # 'Logistic Regression' : {},
        'kNN' : {}
    }

    model_selector = ModelSelector(models, params)
    model_selector.fit(X_train_new, y_train_new)
    model_selector.score_summary()

    # Here all_scores is a list of tuples where a tuple is (model_name, score)
    all_scores = list(model_selector.scores.items())
    # We then sort these tuples based on descending score, then index into this sorted list to pull out the model name of the highest scoring model.
    best_model_name = sorted(all_scores, key=lambda tup : tup[1], reverse=True)[0][0]
    # Then we're pulling out the grid searched model with the highest score to then pickle. Note: to get the actual model instance, call .best_estimator_ on this best_gs_model.
    best_gs_model = model_selector.grid_searches[best_model_name].best_estimator_

    # pickle the model
    with open('data/model_tyler_test.pkl', 'wb') as f:
        pickle.dump(best_gs_model, f)
