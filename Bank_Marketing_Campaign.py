# import libraries
import joblib
import pandas as pd
import streamlit as st


# load the label encoder, one-hot encoder, standard scaler, and model
label_encoder = joblib.load('label_encoder.joblib')
one_hot_encoder = joblib.load('one_hot_encoder.joblib')
scaler = joblib.load('scaler.joblib')
final_model = joblib.load('final_model.joblib')


st.set_page_config(page_title='Customer Churn', layout='wide', initial_sidebar_state='expanded')


st.title('Customer Churn Prediction')

# introduction
st.write('Welcome to the customer churn prediction web application! Churn, or the cancellation of a service by a customer, '
         'is a significant problem in the telecommunications industry. It can lead to lost revenue and decreased '
         'profitability for companies. This web application is designed to help telecommunications companies identify '
         'customers who are at risk of churning, so that they can take action to try to retain those customers.')
st.write('By analyzing data on customer demographics, service usage, billing information, and customer behaviour, '
         'our application can provide a prediction of churn risk for each individual customer. This can help companies '
         'prioritize their retention efforts and allocate resources more effectively.')
st.write('---')


# data input  and output
st.write('To make a churn prediction, the web application requires the following types of data: ')
st.markdown('**Customer demographic**: This includes information such as age, gender, marital status, dependents, '
            'number of dependents, and location.')
st.markdown('**Service usage data**: This application needs data on the type of phone service and internet service that '
            'customers have, as well as their data service and any additional online services they use (such as online '
            'security, online backup, device protection plan, premium technical support, and streaming services).')
st.markdown('**Billing information**: This application also need to know about customer\'s billing and payment preferences, '
            'including their contract type, paperless billing status, payment method, monthly charges, and total charges. '
            'This application also needs data on customers\' long distance usage, including average monthly charges and '
            'total charges, and data usage, including average monthly data usage and any additional data charges.')
st.markdown('**Additional information**: This application also requires data on customers\' tenure with the company '
            '(in months), their satisfaction score, customer lifetime value (CLTV), and whether they have referred a '
            'friend or received referrals.')
st.markdown('**Please make sure to have this data available before using the web application.**')
st.write('After inputting the required data, the web application will provide a churn prediction for each individual '
         'customer. Please note that this prediction is not a guarantee, but is intended to serve as a guide for determining '
         'which customers may require additional attention or retention efforts.')
st.write('---')


# user instruction
st.write('To use the customer churn prediction web application, follow these steps: ')
st.write('1. Input the required data for each customer.')
st.write('2. Click the \'Predict\' button to generate a churn prediction for each customer.')
st.write('3. View the churn predictions.')
st.write('You can use the results of the churn predictions to prioritize your retention efforts and allocate resources '
         'more effectively.')
st.write('---')


st.header('Demographic Information')
# select one of the age group options
age_ranges = st.selectbox('Age Group', ['10-19', '20-29', '30-39', '40-49', '50-59', '60-69', '70-79', '80-89', '90-99'])

# select one of the type of job options
job = st.selectbox('Job', ['housemaid' 'services' 'admin' 'blue-collar' 'technician' 'retired'
                            'management' 'unemployed' 'self-employed' 'unknown' 'entrepreneur'
                            'student'])

# select one of the marital status options
marital = st.selectbox('Marital Status', ['married', 'single', 'divorced', 'unknown'])

# select one of the level of education options
education = st.selectbox('Level of Education', ['basic 4y' 'high school' 'basic 6y' 'basic 9y' 'professional course'
                                                'unknown' 'university degree' 'illiterate'])

# select one of the credit default history option
default = st.selectbox('Credit Default History', ['no', 'unknown', 'yes'])


st.header('Contact Details')
# select one of the preffered means of contact option
contact = st.selectbox('Preffered Means of Contact', ['cellular', 'telephone'])

# select one of the last contact month of the year option
month = st.selectbox('Last Contact Month of the Year', ['mar', 'apr', 'may', 'jun', 'jul',
                                                        'aug', 'sep', 'oct', 'nov', 'dec'])


st.header('Interaction History')
# enter the number of contacts in current campaign
campaign = st.number_input('Number of Contacts in Current Campaign:', min_value=0.0)

# select one of the number of contacts before current campaign options
previous = st.number_input('Number of Contacts Before Current Campaign:', min_value=0.0)
# fit the previous to a group

# select one of the outcome of previous marketing campaign options
poutcome = st.selectbox('Outcome of Previous Marketing Campaign', ['nonexistent', 'failure', 'success'])
if poutcome == 0:
    poutcome = 'never'
elif poutcome == 1:
    poutcome = 'once'
else:
    poutcome = 'multiple times'

st.header('Social and Economic Context Attributed')
# enter the employment variation rate (quarterly indicator)
emp_var_rate = st.number_input('Employment Variation Rate (Quarterly Indicator):', min_value=0.0)

# enter the consumer price index (monthly indicator)
cons_price_idx = st.number_input('Consumer Price Index (Monthly Indicator):', min_value=0.0)

# enter the consumer confidence index (daily indicator)
cons_conf_idx = st.number_input('Consumer Confidence Index (Daily Indicator):', min_value=0.0)

# enter the euribor 3 month rate (daily indicator)
euribor3m = st.number_input('Euribor 3 Month Rate (Daily Indicator):', min_value=0.0)

# enter the number of employees
nr_employed = st.number_input('Number of Employees:', min_value=0.0)


# create an empty data frame
input_df = pd.DataFrame()


# add the user input to the dataframe as a new row
input_df = input_df.append({'age_ranges': age_ranges, 'job': job, 'marital': marital, 'education': education, 'default': default,
                            'contact': contact, 'month': month,
                            'campaign': campaign, 'previous': previous, 'poutcome': poutcome,
                            'emp_var_rate': emp_var_rate, 'cons_price_idx': cons_price_idx, 'cons_conf_idx': cons_conf_idx, 'euribor3m': euribor3m, 'nr_employed': nr_employed}, ignore_index=True)


categorical_columns = ['age_ranges', 'job', 'marital', 'education', 'default', 'contact', 'month', 'previous', 'poutcome']
numerical_columns = ['campaign', 'emp_var_rate', 'cons_price_idx', 'cons_conf_idx', 'euribor3m', 'nr_employed']

# transform the test set
X_test_encoded = one_hot_encoder.transform(X_test[categorical_columns])
X_test_encoded_df = pd.DataFrame(X_test_encoded, columns=one_hot_encoder.get_feature_names_out(categorical_columns))

# transform the test set
X_test_scaled = scaler.transform(X_test[numerical_columns])
X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=numerical_columns)

# combine the test sets
X_test_transformed = pd.concat([X_test_encoded_df, X_test_scaled_df], axis=1)


if st.button('Predict'):
    # make prediction on the data
    result = final_model.predict(X_test_transformed)

    if result == 1:
        st.error('The customer will leave the company.')
    else:
        st.success('The customer will remain with the company.')