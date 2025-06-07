import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder,OneHotEncoder,OrdinalEncoder,PolynomialFeatures
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

def get_numerical_cols(data_features):
    '''
    This method can be called for getting numerical columns from the given dataframe. \n
    takes dataframe as an argument and returns index of columns. 
    '''
    return [col for col in data_features.columns if data_features[col].dtype=='number' or col not in get_categorical_cols(data_features)]
    
def get_categorical_cols(data_features):
    '''
    This method can be called for getting categorical columns from the given dataframe. \n
    takes dataframe as an argument and returns index of columns. 
    '''
    return [col for col in data_features.columns if data_features[col].dtype=='object' or data_features[col].nunique()<10]   

def get_missing_values(data):
    '''
    This method can be called for getting sum of missing values (null values) by column. 
    takes dataframe as an argument and returns summation of all null values in each columns. 
    '''
    return data.isnull().sum()

def drop_null_values(data):
    '''
    This method can be called for deleting (dropping) all the rows with null values. column will stay intact.
    takes dataframe as an argument and returns dataframe without any null values. 
    '''
    return data.dropna()

def del_cols(data, cols):
    '''
    This method can be called for deleting columns if any columns are irrelevant and not needed. 
    takes dataframe and column to be deleted and returns dataframe without the column(s) specified as an argument. 
    '''
    return data.drop(cols, axis =1)

def get_col_names(data):
    return data.columns

def convert_dtype(data,convert_to):
    if convert_to == 'numeric':
        return pd.to_numeric(data,errors='coerce')
    elif convert_to == 'datetime':
        return pd.to_datetime(data)
    elif convert_to == 'dataframe':
        return pd.DataFrame(data)

def get_num_transformer():
    '''
    This method can be called for transformation of numeric columns by imputing null values (as median) and 
    scaling with standard scaler so that data can be scaled. 
    should not pass anything, it returns pipeline with imputed values as median and scaled values so that model will be efficient.
    '''
    return Pipeline(steps=[('Imputer',SimpleImputer(strategy='median')),('scaler',StandardScaler())])

def get_cat_transformer():
    '''
    This method can be called for transformation on categorical columns by imputing null values (as most frequent) and encoding
    with one hot encoder. 
    should not pass anything, it returns pipeline with imputed values as median and scaled values so that model will be efficient.
    '''
    return Pipeline(steps=[('Imputer',SimpleImputer(strategy='most_frequent')),('Encoder',OneHotEncoder(handle_unknown='ignore',sparse_output=False))])

def get_num_poly_transformer(degree =3):
    '''
    This method can be called for polynomial transformation of numeric columns (with degree 3) by imputing null values (as median) and 
    scaling with standard scaler so that data can be scaled. 
    should not pass anything, it returns pipeline with imputed values as median and scaled values so that model will be efficient.
    '''
    return Pipeline(steps=[('poly',PolynomialFeatures(degree=3,include_bias=False)),('scaler',StandardScaler())])

def get_df_from_others_eith_column_names(data,columns = None): # type: ignore
    '''
    This method can convert your series and array type to dataframe.
    '''
    columns = get_col_names(data)
    return pd.DataFrame(data=data,columns=columns)

def get_scatter_plot(data,x,y):
    '''
    This method can be called for plotting scatter graph between two continuous variables. 
    takes dataframe, x component (columns) and y component (columns) and draw scatterplot between x and y columns. 
    '''
    plt.figure(figsize=(20,8))
    sns.scatterplot(data,x=x,y=y)
    plt.show()

def get_kde_plot(data,x,y):
    '''
    This method can be called for plotting density graph between two continuous variables. 
    takes dataframe, x component (columns) and y component (columns) and draw density plot between x and y columns. 
    '''
    plt.figure(figsize=(20,8))
    sns.kdeplot(data=data, x=x,y=y,fill=True)
    plt.show()

def get_heatmap(data):
    '''
    This method can be called for finding correlation between numeric variables. 
    '''
    plt.figure(figsize=(20,8))
    correlation = data.corr()
    sns.heatmap(correlation,annot = True,fmt=".2f",cmap='coolwarm')
    plt.title("Heatmap")
    plt.xticks(rotation=45)
    plt.yticks(rotation=45)
    plt.tight_layout()
    plt.show()  
    
def get_lmplot(data,feature_numeric_data,target_column):
    '''
    This method can be called for plotting regression graph between two continuous variables. 
    takes dataframe, features(columns) and target(columns) and draw regression plots between all features and target one by one. . 
    '''
    for i in feature_numeric_data:
        sns.lmplot(data=data,x=i,y=target_column)
        plt.show()

def get_pairgrid(data,feature_data,target_data):
    '''
    This method can be called for plotting different pairs of variables and target.
    '''
    plt.figure(figsize=(20,8))
    g=sns.PairGrid(data=data,x_vars=feature_data,y_vars=target_data)
    g.map(sns.regplot)
    plt.show()

def get_countplot(data,feature_data,hue):
    '''
    This method can be called for plotting count graph from feature and hue arguments.
    it is for categorical features and categorical target. 
    '''
    plt.figure(figsize=(20,8))
    for i,el in enumerate(feature_data):
        sns.countplot(data=data,x=el,hue=hue)
        plt.show()

def get_pointplot(data,feature_data,target_data):
    '''
    plotting point plot for 
    categorical feature and continuous target
    '''
    sns.pointplot(data,x=feature_data,y=target_data)

def get_boxplot(data,target,feature):
    '''
    plotting box plot for 
    continuous feature and categorical target or
    categorical feature and continuous target
    please pass "x={enter your target variable.values} and y={enter your feature variable.values}" format
    as it is default behaviour of boxplot
    '''
    sns.boxplot(data,x=target,y=feature)