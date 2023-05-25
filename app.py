#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# Function for Churn Prediction
def churn_prediction(input_data):
    loaded_model = pickle.load(open('telecom_churn_trained_model.sav', 'rb'))
    input_data_reshaped = np.asarray(input_data).reshape(1, -1)
    prediction = loaded_model.predict(input_data_reshaped)
    
    if prediction[0] == 0:
        return 'This person is not likely to churn.'
    else:
        return 'This person is likely to churn.'

# Function for Model Evaluation
def evaluate_model(model_name):
    model_metrics = {
        'Logistic Regression': [0.86, 0.85, 0.17],
        'Gaussian NB': [0.86, 0.85, 0.00],
        'Decision Tree': [0.87, 0.87, 0.15],
        'Random Forest': [0.86, 0.85, 0.41],
        'XG Boosting': [1.00, 0.96, 0.85],
        'Support Vector Machine': [1.00, 0.97, 0.90]
    }
    
    if model_name in model_metrics:
        return model_metrics[model_name]
    else:
        return []

# Function for Model Testing
def test_model(classifier_name, params):
    def get_dataset():
        data = pd.read_csv('telecommunications_churn.csv')
        X = data.iloc[:, 0:18]
        y = data.iloc[:, 18]
        return X, y

    def get_classifier():
        if classifier_name == 'SVM':
            clf = SVC(C=params['C'])
        elif classifier_name == 'KNN':
            clf = KNeighborsClassifier(n_neighbors=params['K'])
        elif classifier_name == 'XGboost':
            clf = XGBClassifier(n_estimators=params['n_estimators'], max_depth=params['max_depth'], random_state=1234)
        else:
            clf = RandomForestClassifier(n_estimators=params['n_estimators'], max_depth=params['max_depth'], random_state=1234)
        return clf

    X, y = get_dataset()
    clf = get_classifier()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    return accuracy

# Function for Data Visualization
def visualize_data(choice):
    if choice == 'Exploratory Data Analysis':
        st.subheader('Exploratory Data Analysis')
        data = st.file_uploader('Upload a Dataset', type=['csv', 'txt', 'xlsx'])
        
        if data is not None:
            df = pd.read_csv(data)
            st.dataframe(df.head())
            
            if st.checkbox('Show Shape'):
                st.write(df.shape)
            
            if st.checkbox('Show Columns'):
                all_columns = df.columns.tolist()
                st.write(all_columns)

            if st.checkbox('Select Columns To Show'):
                selected_columns = st.multiselect('Select Columns', all_columns)
                new_df = df[selected_columns
                                new_df = df[selected_columns]
                st.dataframe(new_df)

            if st.checkbox('Show Summary'):
                st.write(df.describe())

            if st.checkbox('Show Value Counts'):
                column = st.selectbox('Select a Column', all_columns)
                value_counts = df[column].value_counts()
                st.write(value_counts)

    elif choice == 'Data Visualization':
        st.subheader('Data Visualization')
        data = st.file_uploader('Upload a Dataset', type=['csv', 'txt', 'xlsx'])

        if data is not None:
            df = pd.read_csv(data)
            st.dataframe(df.head())

            if st.checkbox('Correlation Plot (Seaborn)'):
                st.write(sns.heatmap(df.corr(), annot=True))
                st.pyplot()

            if st.checkbox('Pie Chart'):
                all_columns = df.columns.tolist()
                column_to_plot = st.selectbox('Select 1 Column', all_columns)
                pie_plot = df[column_to_plot].value_counts().plot.pie(autopct="%1.1f%%")
                st.write(pie_plot)
                st.pyplot()

            if st.checkbox('Count Plot'):
                all_columns = df.columns.tolist()
                column_to_plot = st.selectbox('Select 1 Column', all_columns)
                sns.countplot(df[column_to_plot])
                st.write(plt.xticks(rotation=90))
                st.pyplot()

            if st.checkbox('Customizable Plot'):
                st.markdown('### Customizable Plot')
                all_columns_names = df.columns.tolist()
                type_of_plot = st.selectbox('Select Type of Plot', ['area', 'bar', 'line', 'hist', 'box', 'kde'])
                selected_columns_names = st.multiselect('Select Columns To Plot', all_columns_names)

                if st.button('Generate Plot'):
                    st.success(f"Generating Customizable Plot of {type_of_plot} for {selected_columns_names}")
                            if type_of_plot == 'area':
                        cust_data = df[selected_columns_names]
                        st.area_chart(cust_data)
                    elif type_of_plot == 'bar':
                        cust_data = df[selected_columns_names]
                        st.bar_chart(cust_data)
                    elif type_of_plot == 'line':
                        cust_data = df[selected_columns_names]
                        st.line_chart(cust_data)
                    elif type_of_plot == 'hist':
                        cust_data = df[selected_columns_names]
                        plt.hist(cust_data, bins=20, alpha=0.7)
                        st.pyplot()
                    elif type_of_plot == 'box':
                        cust_data = df[selected_columns_names]
                        st.boxplot(cust_data)
                    elif type_of_plot == 'kde':
                        for col in selected_columns_names:
                            plt.figure(figsize=(12, 6))
                            sns.kdeplot(df[col], shade=True)
                            st.pyplot()
                            elif choice == 'Model Evaluation':
                            st.subheader('Model Evaluation')
                            model_name = st.selectbox('Select Model', ['Logistic Regression', 'Gaussian NB', 'Decision Tree', 'Random Forest', 'XG Boosting', 'Support Vector Machine'])
                            result = evaluate_model(model_name)\
                            if result:
                            st.write('Accuracy:', result[1])
                            st.write('Precision:', result[2])
                            st.write('Recall:', result[3])
                            else:
                            st.write('Please select a valid model.')
                            elif choice == 'Model Testing':
                            st.subheader('Model Testing')
        classifier_name = st.selectbox('Select Classifier', ['SVM', 'KNN', 'Random Forest', 'XGboost'])
        
        if classifier_name == 'SVM':
            C = st.number_input('C (Regularization Parameter)', 0.01, 10
        if classifier_name == 'SVM':
            C = st.number_input('C (Regularization Parameter)', 0.01, 10.0, step=0.01, value=1.0)
                                elif classifier_name == 'KNN':
                                K = st.number_input('K (Number of Neighbors)', 1, 20, step=1, value=5)
                                elif classifier_name == 'XGboost':
                                n_estimators = st.number_input('n_estimators (Number of Estimators)', 100, 1000, step=100, value=200)
                                max_depth = st.number_input('max_depth (Maximum Depth)', 1, 10, step=1, value=3)
                                else:
                                n_estimators = st.number_input('n_estimators (Number of Estimators)', 100, 1000, step=100, value=200)

                              max_depth = st.number_input('max_depth (Maximum Depth)', 1, 10, step=1, value=3)

