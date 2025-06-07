import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_selection import RFECV,SequentialFeatureSelector
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression,LinearRegression
from sklearn.cross_decomposition import PLSRegression
import matplotlib.pyplot as plt
import eda
import train as t

#Creating application title
st.title("Feature Selection algorithms")

#Creating file uploader
uploaded_file = st.file_uploader("Enter any dataset to upload (in CSV format please!)",type='csv')

if uploaded_file:
    #creating data frame
    df = pd.read_csv(uploaded_file)
    
    # Preview of data
    st.write("Preview of your data")
    st.dataframe(df.head())

    #Shwing "df.info()" type structure in streamlit app
    if st.checkbox("want to see information (not null values, column names and datatypes of columns) about your data"):
         summary_df = pd.DataFrame({'columns':df.columns,'Non null counts':df.notnull().sum(),'data type':df.dtypes.astype(str)})
         st.dataframe(summary_df)

    #Asking user if any of the columns needed to be converted into numeric
    convert_column = st.selectbox("Select column to convert to numeric",options=['--select a column--']+list(df.columns))
    if convert_column!= '--select a column--':
        df[convert_column] = eda.convert_dtype(df[convert_column],'numeric')

    #asking user based target columns
    target_column = st.selectbox("Select target column",options=['--select a column--']+list(df.columns))
    if target_column != '--select a column--':
        st.info(f"column {target_column} has been selected as target")
    
    #Asking user if any columns can be deleted. (Multiple columns can be deleted)
    deletion_columns = st.multiselect("Select any column to delete (Can select multiple)",
                                     options=['--select a column--']+ list(df.columns))

    #creating backup dataframe
    df_new = df.copy()

    #deleting column(s) based on user input
    if deletion_columns != '--select a column--':
        df_new = eda.del_cols(df,deletion_columns)
        st.success(f"Deleted Columns ,{''.join(deletion_columns)} ")
    else:
        st.info("You have not selected any column to be deleted")

    #removing null values
    remove_null = st.checkbox("If you want to remove null values")
    if remove_null:
        df_new = eda.drop_null_values(df_new)

    #asking user which method they want to do the feature selection
    method = st.selectbox("Enter your choice of feature selection method",
                 ('--Select a method--','Recursive Feature engineering(RFE)','Sequential Feature Selector','PCA','LDA')
                 ) 

    #Creating X and y variables so that training and testing dataset can be created.
    X = df_new.drop(target_column,axis=1)    
    if st.checkbox("Whether target column needed to be mapped as 0 and 1?"):
        df_new[target_column] = df_new[target_column].map({df_new[target_column].value_counts().index[1]:1,
                                                           df_new[target_column].value_counts().index[0]:0})
    y = df_new[target_column]

    #finding out categorical and numerical columns so that scaling/encoding can happen basis on user requirement.
    cat_cols = eda.get_categorical_cols(X)
    num_cols = eda.get_numerical_cols(X)
    if st.checkbox("want to see Numerical Columns?"):
         st.write(list(num_cols))
    if st.checkbox("want to see Categorical Columns?"):
         st.write(list(cat_cols))

    #Encoding of categorical columns
    if cat_cols:
         X_encoded = pd.get_dummies(X[cat_cols],drop_first=True)
         st.write("Encoding of categorical column has been done")
         if num_cols:
              X_num = X[num_cols]
         X = pd.concat([X_encoded,X_num],axis=1)
    else:
         X = X[num_cols]
    
    #We can show columns that are going into train test split
    if st.checkbox("want to see Total columns to go into train test split?"):
         st.write(X.columns)

    #splitting into train and test data
    X_train,X_test,y_train,y_test = t.get_train_test_split_data(X,y)

    #User can check if they want to see null values on training data.
    if st.checkbox("want to see null values from training dataset?"):
         st.write("X_train null values",X_train.isnull().sum())
         st.write("y_train null values",y_train.isnull().sum())
    
    #creating out of box logistic regression model
    list_models = ['Logistic Regression','regression']
    model_name=st.selectbox("Please select which model do you want to use:" ,list_models)
    if model_name == 'Logistic Regression':
        model = LogisticRegression(penalty='l1',solver = 'liblinear',max_iter=300)
    elif model_name == 'regression':
         model = LinearRegression()

    if method == 'Recursive Feature engineering(RFE)':
        # Creating RFE
        rfe = RFECV(model,min_features_to_select=1,cv=5)

        # creating training and test data transformation
        X_train_transformed = rfe.fit_transform(X_train,y_train)
        X_test_transformed = rfe.transform(X_test)
        X_train_rfe = pd.DataFrame(X_train_transformed)
        X_test_rfe = pd.DataFrame(X_test_transformed)

        #Show features
        selected_features = X_train.columns[rfe.support_]
        if st.checkbox("want to see Selected features?"):
             st.write(list(selected_features))

        #fitting the model
        model.fit(X_train_rfe,y_train)
        rfe2_pred = model.predict(X_test_rfe)

        #writing the accuracy
        st.write("Evaluation metrices of the model is: ")
        st.write(t.evaluate_model(y_test,rfe2_pred,model_name))

        #plotting the graph
        st.subheader("Accuracy vs. number of feature selectors")
        fig,ax = plt.subplots()
        ax.plot(range(1,len(rfe.cv_results_['mean_test_score'])+1),rfe.cv_results_['mean_test_score'],marker='o')
        ax.set_title("RFECV feature selection curve")
        ax.set_xlabel("Number of selected features")
        ax.set_ylabel("accuracy")
        st.pyplot(fig)

    elif method =='Sequential Feature Selector':
        #Enter user based number of features slider
        num_features = st.slider("Please select features",1,min(X_train.shape[1],50),5)

        #Creating SFS model 
        sfs = SequentialFeatureSelector(model,n_features_to_select=num_features,direction='forward')

        #Transforming on training and testing data
        X_train_transformed = sfs.fit_transform(X_train,y_train)
        X_test_transformed = sfs.transform(X_test)

        #transforming into dataframe with slected features so that new accuracy can be looked. 
        X_train_sfs = pd.DataFrame(X_train_transformed,columns = sfs.get_feature_names_out())
        X_test_sfs = pd.DataFrame(X_test_transformed,columns = sfs.get_feature_names_out())
        
        #Show selected features in a streamlit app
        selected_features = X_train.columns[sfs.get_support()]
        if st.checkbox("want to see Selected features?"):
             st.write(list(selected_features))

        #fitting the model
        model.fit(X_train_sfs,y_train)
        y_pred_sfs = model.predict(X_test_sfs)

        #writing accuracy in streamlit app
        st.write("Evaluation metrices of the model is: ")
        st.write(t.evaluate_model(y_test,y_pred_sfs,model_name))

        # plotting feature ranking.
        if st.checkbox("Want to see feature ranking i.e. features have been selected in which order?"):
             fig,ax = plt.subplots()
             ax.barh(selected_features,range(len(selected_features)))
             ax.invert_yaxis()
             ax.set_title("Feature ranking")
             ax.set_xlabel("Selection order by ranking")
             st.pyplot(fig)

    elif method == 'PCA':
        #Asking user to select number of features
        num_features = st.slider("Enter number of features to see accuracy on: ",1,X_train.shape[1],5)

        #creating PCA
        pca = PCA(n_components=num_features)

        #fitting on training and testing data
        X_train_transformed = pca.fit_transform(X_train)
        X_test_transformed = pca.transform(X_test)
        
        #this can show variance ratio
        if st.checkbox("Whether you want to see the variance ratio?"):
                         st.write(pca.explained_variance_ratio_)

        #fitting the model
        model.fit(X_train_transformed,y_train)

        #Predicting the target
        y_pred_pca = model.predict(X_test_transformed)

        #writing the accuracy
        st.write("Evaluation metrices of the model is: ")
        st.write(t.evaluate_model(y_test,y_pred_pca,model_name))

        #Visualizing based on user input
        if st.checkbox("You want to visualize variance explanation by components?"):
             explain_variances_ratio = pca.explained_variance_ratio_
             cumulative_variance = np.cumsum(explain_variances_ratio)

             fig,ax = plt.subplots()
             ax.plot(range(1,len(cumulative_variance)+1),cumulative_variance)
             ax.set_title("Cumulative variance explained by PCA components")
             ax.set_xlabel("PCA components")
             ax.set_ylabel("Cumulative explained variance")
             ax.grid(True)
             st.pyplot(fig)

    elif method == 'LDA':
         #creating LDA
         lda = LinearDiscriminantAnalysis()

         #fitting LDA on training and testing data
         if model_name == 'Logistic Regression':
            X_train_lda = lda.fit_transform(X_train,y_train)
            X_test_lda = lda.transform(X_test)
            #fetching and showing selected features
            selected_features = lda.get_feature_names_out()
            st.write("Selected_features")
            st.write(list(selected_features))
            #fitting the model and predicting target from model
            model.fit(X_train_lda,y_train)
            y_pred_lda = model.predict(X_test_lda)
         else:
            pls = PLSRegression(n_components=2)
            pls.fit(X_train,y_train) 
            #fetching and showing selected features
            selected_features = pls.get_feature_names_out()
            st.write("Selected_features")
            st.write(list(selected_features))    
            #fitting the model and predicting target from model
            model.fit(X_train,y_train)
            y_pred_lda = model.predict(X_test)

         #printing the accuracy
         st.write("Evaluation metrices of the model is: ")
         st.write(t.evaluate_model(y_test,y_pred_lda,model_name))
         
    else:
         st.write("Enter correct method name from the available feature selection methods/algorithms.")    
    st.success("Feature selection completed")