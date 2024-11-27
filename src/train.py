import joblib
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

def train_and_save_models(X_train, y_train):
    """
    Train multiple models and save them to the models/ folder.
    """
    # SVM
    param_grid_svm = {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']}
    svm = GridSearchCV(SVC(), param_grid_svm, cv=3)
    svm.fit(X_train, y_train)
    joblib.dump(svm.best_estimator_, 'models/svm_model.pkl')

    # Logistic Regression
    param_grid_logreg = {'C': [0.1, 1, 10], 'solver': ['liblinear']}
    logreg = GridSearchCV(LogisticRegression(), param_grid_logreg, cv=3)
    logreg.fit(X_train, y_train)
    joblib.dump(logreg.best_estimator_, 'models/logreg_model.pkl')

    # Random Forest
    param_grid_rf = {'n_estimators': [50, 100], 'max_depth': [10, None]}
    rf = GridSearchCV(RandomForestClassifier(), param_grid_rf, cv=3)
    rf.fit(X_train, y_train)
    joblib.dump(rf.best_estimator_, 'models/rf_model.pkl')

    # XGBoost
    param_grid_xgb = {'max_depth': [3, 5], 'learning_rate': [0.1, 0.01], 'n_estimators': [100, 200]}
    xgb_model = GridSearchCV(xgb.XGBClassifier(use_label_encoder=False), param_grid_xgb, cv=3)
    xgb_model.fit(X_train, y_train)
    joblib.dump(xgb_model.best_estimator_, 'models/xgb_model.pkl')
