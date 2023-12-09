import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns 
import plotly.express as px
import time
import streamlit as st

def set_custom_style():
    st.markdown("""
        <style>
        div[role="main"] { background-color: #191970; color: #FFFFFF; }
        div[data-testid="stSidebar"], div[data-testid="stSidebar"] .st-bb { background-color: #F5F5F7; color: #333333; }
        button { background-color: #007AFF; color: #FFFFFF; }
        input, select { background-color: #E5E5EA; color: #333333; }
        h1, h2, h3, h4, h5, h6, div[data-testid="stSidebar"] h1, div[data-testid="stSidebar"] h2, div[data-testid="stSidebar"] h3, div[data-testid="stSidebar"] h4, div[data-testid="stSidebar"] h5, div[data-testid="stSidebar"] h6 { color: #FFFFFF; }
        </style>
        """, unsafe_allow_html=True)

set_custom_style()

# extract values for dropdowns
def extract_unique_values(df, column):
    return df[column].dropna().unique().tolist()

# Loading models
with open('decision_tree_model.pkl', 'rb') as file:
    decision_tree_model = pickle.load(file)
with open('regression.pkl', 'rb') as file:
    regression_model = pickle.load(file)

employability_data = pd.read_csv('model_data.csv')
salary_data = pd.read_csv('processed_salary_data.csv')
full_data = pd.read_csv('stackoverflow_full.csv')

# Gte min-max of numerical columns for inout normalization
skills_list_employability = [col for col in employability_data.columns if not col.startswith(('Country_', 'Age', 'Accessibility', 'EdLevel', 'Employment', 'Gender', 'MentalHealth', 'MainBranch', 'YearsCode', 'YearsCodePro', 'ComputerSkills', 'Employed','PreviousSalary'))]
min_max_values_employability = {
    'YearsCode': (full_data['YearsCode'].min(), full_data['YearsCode'].max()),
    'YearsCodePro': (full_data['YearsCodePro'].min(), full_data['YearsCodePro'].max()),
    'PreviousSalary': (full_data['PreviousSalary'].min(), full_data['PreviousSalary'].max()),
    'ComputerSkills': (full_data['ComputerSkills'].min(), full_data['ComputerSkills'].max())
}

# Define skills list and min-max values for salary prediction
skills_list_salary = [col for col in salary_data.columns if not col.startswith(('Country_', 'EdLevel', 'Gender', 'YearsCode', 'YearsCodePro', 'ComputerSkills', 'Employed','PreviousSalary'))]
min_max_values_salary = {
    'YearsCode': (full_data['YearsCode'].min(), full_data['YearsCode'].max()),
    'YearsCodePro': (full_data['YearsCodePro'].min(), full_data['YearsCodePro'].max()),
    'ComputerSkills': (full_data['ComputerSkills'].min(), full_data['ComputerSkills'].max())
}

# Process user inputs checking employee suitability
def process_input_employability(user_data):
    input_df = pd.DataFrame([user_data])
    for skill in skills_list_employability:
        input_df[skill] = 1 if skill in user_data['Skills'] else 0
    
    #normalization
    for col, (min_val, max_val) in min_max_values_employability.items():
        input_df[col] = (input_df[col] - min_val) / (max_val - min_val)
    #one-hot encoding
    categorical_cols = ['Age', 'Accessibility', 'EdLevel', 'Employment', 'Gender', 'MentalHealth', 'MainBranch', 'Country']
    input_df = pd.get_dummies(input_df, columns=categorical_cols, prefix=categorical_cols)

    final_df = pd.DataFrame(columns=[col for col in employability_data.columns if col != 'Employed'])
    for col in final_df.columns:
        final_df[col] = input_df[col] if col in input_df else 0
    return final_df

# process inputs for salary estimation
def process_input_salary(user_data):
    input_df = pd.DataFrame([user_data])

    for skill in skills_list_salary:
        input_df[skill] = 1 if skill in user_data['Skills'] else 0
    # normalization
    for col, (min_val, max_val) in min_max_values_salary.items():
        input_df[col] = (input_df[col] - min_val) / (max_val - min_val)

    # One-hot encoding
    categorical_cols = ['EdLevel', 'Gender', 'Country']
    input_df = pd.get_dummies(input_df, columns=categorical_cols, prefix=categorical_cols)

    final_df = pd.DataFrame(columns=[col for col in salary_data.columns if col != 'PreviousSalary'])
    for col in final_df.columns:
        final_df[col] = input_df[col] if col in input_df else 0
    return final_df

st.sidebar.title("Navigation")
page = st.sidebar.radio("Choose the Application", ["Employee Suitability Check", "Salary Estimation"])

# Employability Suitability Check
if page == "Employee Suitability Check":
    st.title('Employee Suitability Check')

    with st.form("Employee Suitability Check Form"):
        age = st.selectbox('Age', extract_unique_values(full_data, 'Age'))
        accessibility = st.selectbox('Accessibility', extract_unique_values(full_data, 'Accessibility'))
        ed_level = st.selectbox('Education Level', extract_unique_values(full_data, 'EdLevel'))
        employment = st.selectbox('Employment', extract_unique_values(full_data, 'Employment'))
        gender = st.selectbox('Gender', extract_unique_values(full_data, 'Gender'))
        mental_health = st.selectbox('Mental Health', extract_unique_values(full_data, 'MentalHealth'))
        main_branch = st.selectbox('Main Branch', extract_unique_values(full_data, 'MainBranch'))
        country = st.selectbox('Country', extract_unique_values(full_data, 'Country'))
        years_code = st.slider('Years of Coding', 0, 50, 5)
        years_code_pro = st.slider('Years of Professional Coding', 0, 50, 5)
        previous_salary = st.number_input('Previous Salary')
        computer_skills = st.slider('Computer Skills', 1, 10, 5)
        skills = st.multiselect('Technologies Worked With', skills_list_employability)

        submitted = st.form_submit_button("Submit Application")

    if submitted:
        st.session_state['employability_submitted'] = True
        progress_bar = st.progress(0)
        for i in range(100):
            time.sleep(0.01)
            progress_bar.progress(i + 1)

        user_data = {
            'Age': age,
            'Accessibility': accessibility,
            'EdLevel': ed_level,
            'Employment': employment,
            'Gender': gender,
            'MentalHealth': mental_health,
            'MainBranch': main_branch,
            'Country': country,
            'YearsCode': years_code,
            'YearsCodePro': years_code_pro,
            'PreviousSalary': previous_salary,
            'ComputerSkills': computer_skills,
            'Skills': skills
        }
        processed_input = process_input_employability(user_data)
        employability = decision_tree_model.predict(processed_input)[0]
        st.write('Employee Suitability Check:', 'Candidate is Suitable' if employability == 1 else 'Candidate is Not Suitable')
        st.subheader('Years of Coding by Age and Gender')
        fig1, ax1 = plt.subplots()
        sns.violinplot(x='Age', y='YearsCode', hue='Gender', data=full_data, ax=ax1)
        ax1.set_title('Years of Coding by Age and Gender')
        ax1.set_xlabel('Age')
        ax1.set_ylabel('Years of Coding')
        ax1.tick_params(axis='x', rotation=45)
        st.pyplot(fig1)

        # EDA: Years of Coding Experience by Education Level
        st.subheader('Years of Coding Experience by Education Level')
        fig2, ax2 = plt.subplots()
        sns.boxplot(x="EdLevel", y="YearsCode", data=full_data, ax=ax2)
        ax2.set_title('Years of Coding Experience by Education Level')
        ax2.set_xlabel('Education Level')
        ax2.set_ylabel('Years of Coding')
        ax2.tick_params(axis='x', rotation=45)
        st.pyplot(fig2)

    if st.session_state.get('employability_submitted', False):
        st.subheader('Dynamic Graphs')

        # Dynamic graphs
        option_1 = st.selectbox('Select the first parameter:', ['Age', 'Gender', 'EdLevel'], key='dynamic_option_1')
        option_2 = st.selectbox('Select the second parameter:', ['PreviousSalary', 'YearsCode', 'YearsCodePro', 'ComputerSkills'], key='dynamic_option_2')

        if st.button('Generate Visualization'):
            if option_1 in full_data and option_2 in full_data:
                plt.figure(figsize=(10, 6)) 
                sns.violinplot(data=full_data, x=option_1, y=option_2)
                plt.title(f'{option_1} vs {option_2} (Violin Plot)')
                plt.xlabel(option_1)
                plt.ylabel(option_2)
                st.pyplot(plt) 
            else:
                st.error("Invalid column selection. Please select different parameters.")

# Salary Estimation Page
elif page == "Salary Estimation":
    st.title('Salary Estimation')

    with st.form("salary_estimation_form"):
        ed_level = st.selectbox('Education Level', ['Master', 'NoHigherEd', 'Other', 'PhD', 'Undergraduate'])
        gender = st.selectbox('Gender', ['Man', 'NonBinary', 'Woman', 'Other'])
        country = st.selectbox('Country', extract_unique_values(full_data, 'Country'))
        years_code = st.slider('Years of Coding', 0, 50, 5)
        years_code_pro = st.slider('Years of Professional Coding', 0, 50, 5)
        computer_skills = st.slider('Computer Skills', 1, 10, 5)
        skills = st.multiselect('Technologies Worked With', skills_list_salary)
        employed = st.selectbox('Employed', [0, 1])

        submitted = st.form_submit_button("Submit Salary Estimation")

    if submitted:
        progress_bar = st.progress(0)
        for i in range(100):
            time.sleep(0.01)
            progress_bar.progress(i + 1)
        user_data = {
            'YearsCode': years_code,
            'YearsCodePro': years_code_pro,
            'ComputerSkills': computer_skills,
            'EdLevel': f'EdLevel_{ed_level}',
            'Gender': f'Gender_{gender}',
            'Country': country,
            'Skills': skills,
            'Employed': employed
        }
        all_salaries = salary_data['PreviousSalary'].dropna().values
        processed_input = process_input_salary(user_data)
        predicted_salary = regression_model.predict(processed_input)[0]


        st.write('Salary Prediction:', predicted_salary)

        hist_fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(all_salaries, bins=20, color="lightblue", edgecolor='black', alpha=0.7)
        ax.axvline(predicted_salary, color='red', linestyle='dashed', linewidth=2, label='Predicted Salary')
        ax.set_title("Salary Distribution with Predicted Value")
        ax.set_xlabel("Salary")
        ax.set_ylabel("Frequency")
        ax.legend()
        ax.grid(True)
        st.pyplot(hist_fig) 

        #  plot
        selected_columns = st.multiselect('Select Columns for Pair Plot', salary_data.columns, default=['YearsCode', 'YearsCodePro', 'ComputerSkills', 'PreviousSalary'])
        if len(selected_columns) >= 2:
            st.subheader('Pair Plot of Selected Columns')
            pairplot_fig = sns.pairplot(salary_data[selected_columns], diag_kind='kde')
            st.pyplot(pairplot_fig.fig) 
        else:
            st.write("Please select at least two columns for the pair plot.")

            
            