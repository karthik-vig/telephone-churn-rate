# Telephone-churn-rate
A project to predict whether a customer will leave the telephone company based on the data from past three months.

# Dataset:
The dataset is from a Kaggle competition. Link: https://www.kaggle.com/competitions/telecom-churn-case-study-hackathon-C39

# Feature Extraction:
First the unnecessary columns were removed. These columns are 'id', 'circle_id', 'last_date_of_month_6', 'last_date_of_month_7', 'last_date_of_month_8'.
Next columns such as the 'date_of_last_rech_6', 'date_of_last_rech_7', 'date_of_last_rech_8' were changed into days between recharges. The same process has
been applied for columns 'date_of_last_rech_data_6', 'date_of_last_rech_data_7', 'date_of_last_rech_data_8'. As for the missing data, the mean of a feature was
used for continuous values and the most repeated value was used for categorical feature data.