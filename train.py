import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression,LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier,DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier,AdaBoostRegressor,GradientBoostingClassifier,GradientBoostingRegressor
from sklearn.ensemble import BaggingClassifier,BaggingRegressor
from sklearn.naive_bayes import BernoulliNB
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import mean_absolute_percentage_error, classification_report,mean_squared_error,accuracy_score
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Lasso, Ridge, ElasticNet
from imblearn.over_sampling import SMOTE
import eda as eda

def get_train_test_split_data(feature_df,target_df,test_size =0.25,random_state=42):
    '''
    This method can be called to create train test split and return Xtrain and test data 
    and ytrain and test data.
    '''
    return train_test_split(feature_df,target_df,test_size=test_size,random_state=random_state)

def model_pipeline(feature_df,target_df,num_cols, cat_cols,task,smote=0):
    '''
    Returning model with prediction and testing values after train and test split
    and model fitting on train data and predicting on test data
    '''
    X_train,X_test,y_train,y_test = train_test_split(feature_df,target_df,test_size=0.2,random_state=42)
    model_type = get_task(task)    
    y_train = y_train.values.ravel()
    y_test = y_test.values.ravel()
   

    if smote == 1:
        X_train_raw = X_train.copy()
        num_columns = eda.get_numerical_cols(X_train_raw)
        cat_columns = eda.get_categorical_cols(X_train_raw)
        preprocessor = get_preprocessing(num_columns,cat_columns)
        model = model_type
        X_train_transformed = preprocessor.fit_transform(X_train_raw)
        preprocessor_columns = preprocessor.get_feature_names_out()
        X_train_df = pd.DataFrame(X_train_transformed,columns=preprocessor_columns)
        y_train = pd.Series(y_train).astype(int)
        print(y_train.shape,y_train.dtype,np.unique(y_train))
        X_train_resampled,y_train_resampled = get_resampled_smote(X_train_df,y_train)
        model.fit(X_train_resampled,y_train_resampled)
        X_test_transformed = preprocessor.transform(X_test)
        X_test_df = pd.DataFrame(X_test_transformed,columns=preprocessor_columns)
        
    else:
        model = make_pipeline(get_preprocessing(num_cols,cat_cols),model_type)
        model.fit(X_train,y_train)
    y_pred=model.predict(X_test)    

    if task in['xgboost', 'randon_forest']:    #adding xgboost related code so that it will work smoothly with xgboost. 
        y_test = pd.Series(y_test).astype(int)
    y_pred = pd.Series(y_pred).astype(int)
    return model,y_test,y_pred,X_train,y_train

def get_task(task):
    '''
    This method can be called for getting one model_type from pool of regression and classification models.
    '''
    if task == 'linear_regression':
        model = LinearRegression()
    elif task == 'logistic_regression':
        model = LogisticRegression()
    elif task == 'knn':
        model = KNeighborsClassifier()
    elif task == 'decision_tree':
        model = DecisionTreeClassifier()
    elif task == 'randon_forest':
        model = RandomForestClassifier()
    elif task == 'svm':
        model = SVC()
    elif task == 'naive_bayes':
        model = BernoulliNB()
    elif task == 'Ada Boost classifier':
        model = AdaBoostClassifier(n_estimators=200,estimator=DecisionTreeClassifier())
    elif task == 'Ada Boost regressor':
        model = AdaBoostRegressor(n_estimators=200,estimator=DecisionTreeRegressor())
    elif task == 'Gradient boost classifier':
        model = GradientBoostingClassifier(n_estimators=200)
    elif task == 'Gradient boost regressor':
        model = GradientBoostingRegressor(n_estimators=200)
    elif task == 'Bagging classifier':
        model = BaggingClassifier(estimator=DecisionTreeClassifier(),n_estimators=100,random_state=42)
    elif task == 'Bagging regressor':
        model = BaggingClassifier(estimator=DecisionTreeRegressor(),n_estimators=100,random_state=42)
    elif task == 'xgboost':
        model = XGBClassifier(learning_rate = 0.1, max_depth = 4,enable_categorical = True,n_estimators = 100)
    else:
        print("Kindly enter valid task name; task name includes [linear_regression,logistic_regression,knn," \
        "      decision_tree,randon_forest,svm,naive_bayes, Ada Boost classifier, Ada Boost regressor, " \
        "Gradient boost classifier, Gradient boost regrssor, Bagging regressor, Bagging classifier, xgboost]")
    return model

#using SMOTE.
def get_resampled_smote(X_trainining,y_trainining):
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_trainining,y_trainining)
    return X_resampled,y_resampled

def model_poly_pipeline(feature_df,target_df,num_cols, cat_cols):
    '''
    Returning model with prediction and testing values after train and test split
    and model fitting on train data and predicting on test data - for polynomial transformation.
    '''
    X_train,X_test,y_train,y_test = train_test_split(feature_df,target_df,test_size=0.2,random_state=42)
    model = make_pipeline(get_poly_preprocessing(num_cols,cat_cols),LinearRegression())
    model.fit(X_train,y_train)
    y_pred=model.predict(X_test)
    return model,y_test,y_pred

def get_cols(feature_df):
    '''
    This method can be called for fetching columns categorized into numerical and categorical columns.
    takes argument as dataframe (generally features of dataframe and returns numerical and categorical columns
    in that order)
    '''
    boston_df_num_cols = eda.get_numerical_cols(feature_df)
    boston_df_cat_cols = eda.get_categorical_cols(feature_df)
    return boston_df_num_cols,boston_df_cat_cols

def get_preprocessing(num_cols, cat_cols):
    '''
    This method can be called for returning numerical columns' transformation which uses numerical transformer resides in EDA.py and
    categorical transformation resides in EDA.py from the arguments numerical columns and categorical columns.
    '''
    return ColumnTransformer(transformers=[('num',eda.get_num_transformer(),num_cols),('cat',eda.get_cat_transformer(),cat_cols)])

def get_poly_preprocessing(num_cols,cat_cols):
    '''
    This method can be called for returning numerical columns' transformation which uses numerical transformer resides in EDA.py and
    categorical transformation resides in EDA.py from the arguments numerical columns and categorical columns. This is for 
    polynomial transformation of numerical columns.
    '''
    return ColumnTransformer(transformers=[('num',eda.get_num_poly_transformer(degree=3),num_cols),('cat',eda.get_cat_transformer(),cat_cols)])

def get_MAPE(y_test,y_pred):
    '''
    Returning the mean percentage absolute error of y_test and y_pred from the arguments y_test and y_pred in that order.
    '''
    return mean_absolute_percentage_error(y_test,y_pred)

def get_training_error(feature_df,target_df,num_cols, cat_cols):
    '''
    Returning the training error (mean absolute percentage error) from the arguments features of dataframe, target of dataframe, numerical columns and
    categorical columns after training and testing split and then creating a preprocessing timeline on the same
    with doing linear regression
    '''
    x_train,x_test,y_train,y_test = train_test_split(feature_df,target_df,test_size=0.2,random_state=42)
    model = make_pipeline(get_preprocessing(num_cols,cat_cols),LinearRegression())
    model.fit(x_train,y_train)
    y_train_pred = model.predict(x_train)
    train_mape = mean_absolute_percentage_error(y_train,y_train_pred)*100
    return train_mape

def get_lasso_mape(alpha,features, target):
    '''
    This method can be called for doing regularization with lasso. 
    It takes arguments alpha (0,1,1,10,etc.) , features and target. 
    It returns lasso mean absolute percentage error after generating cross val score.
    '''
    lasso = Lasso(alpha=alpha)
    lasso_mape = cross_val_score(lasso,features,target,scoring='neg_mean_absolute_percentage_error',cv=5).mean()*-100
    return lasso_mape

def get_ridge_mape(alpha,features, target):
    '''
    This method can be called for doing regularization with Ridge. 
    It takes arguments alpha (0,1,1,10,etc.) , features and target. 
    It returns Ridge mean absolute percentage error after generating cross val score.
    '''
    ridge = Ridge(alpha=alpha)
    ridge_mape = cross_val_score(ridge,features,target,scoring='neg_mean_absolute_percentage_error',cv=5).mean()*-100
    return ridge_mape

def get_elasticnet_mape(alpha,features, target):
    '''
    This method can be called for doing regularization with Elastic net. 
    It takes arguments alpha (0,1,1,10,etc.) , features and target. 
    It returns elastic net mean absolute percentage error after generating cross val score.
    '''
    elastic = ElasticNet(alpha=alpha)
    elastic_mape = cross_val_score(elastic,features,target,scoring='neg_mean_absolute_percentage_error',cv=5).mean()*-100
    return elastic_mape

def evaluate_model(y_test,y_pred,task):
    '''
    This method can be called for evaluating model based on task task is regression for linear regression and others for classification.
    it returns MAPE and MSE for linear regression and classification report for classification related datasets.
    '''
    try:
        if task == 'regression':
            return{
            'MAPE': f"MAPE is: {(get_MAPE(y_test,y_pred))*100}",
            'MSE' : mean_squared_error(y_test,y_pred)
            }
        else:
            return{
                'Accuracy Score' : f'{round(accuracy_score(y_test,y_pred)*100,2)}%',
                #'Classification report':print(classification_report(y_test,y_pred))
            }
    except Exception as e:
        print(e, "Enter 'regression' for numerical variables otherwise it will run classification report or gives predicted y value")