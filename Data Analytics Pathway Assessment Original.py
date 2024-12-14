#!/usr/bin/env python
# coding: utf-8

# In[8]:


# Libraries 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc


# In[9]:


# Load and display the dataset
data = pd.read_csv("C:\\Users\\LENOVO\\Downloads\\Smash\\data 1\\data\\bank-additional-full.csv", sep=';')  # Assuming it uses a semicolon as a separator based on the dataset name.
data


# In[10]:


data_info = data.info()
data_info


# In[11]:


# 1. Data preparation and exploratory data analysis (EDA)
# Checking for missing values
missing_values = data.isnull().sum()
missing_values


# In[12]:


# Standardizing column names 
data.columns = data.columns.str.strip().str.lower().str.replace('.', '_')
data.columns


# In[13]:


# Checking for duplicates and removing them
duplicates_count = data.duplicated().sum()
data_cleaned = data.drop_duplicates()
data_cleaned


# In[14]:


# Summary after cleaning
data_cleaned_info = data_cleaned.info()
missing_values, duplicates_count, data_cleaned_info


# In[15]:


# Summary statistics for numerical variables
summary_statistics = data.describe()
summary_statistics


# In[16]:


# Setting the style for the plots
sns.set(style="whitegrid")


# In[23]:


# Plotting histograms for numerical variables
numerical_columns = ['age', 'duration', 'campaign', 'pdays', 'previous', 'emp_var_rate', 'cons_price_idx', 'cons_conf_idx',	'euribor3m', 'nr_employed']
data[numerical_columns].hist(bins=15, figsize=(15, 15), layout=(4, 3), color ='#b4446c', zorder=2, rwidth=1.0)
plt.suptitle('Distribution of Numerical Variables', fontsize=16)
plt.show()


# In[27]:


# Bar plots for categorical variables
categorical_columns = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'day_of_week', 'poutcome', 'y']
fig, axes = plt.subplots(6, 2, figsize=(15, 20))
for i, col in enumerate(categorical_columns):
    sns.countplot(y=col, data=data, ax=axes[i//2, i%2], palette='flare')
    axes[i//2, i%2].set_title(f'Distribution of {col}')
    axes[i//2, i%2].set_xlabel('')
    axes[i//2, i%2].set_ylabel('')
plt.tight_layout()
plt.show()


# In[28]:


# Correlation heatmap for numeric columns
plt.figure(figsize=(10, 6))
correlation_matrix = data[numerical_columns].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap')
plt.show()


# In[29]:


# Checking for outliers using boxplots for numeric columns
plt.figure(figsize=(10, 6))
sns.boxplot(data=data[numerical_columns])
plt.title('Boxplot to Detect Outliers')
plt.xticks(rotation=90)  # Rotate column labels for better visibility
plt.show()


# In[30]:


# 2. Feature Engineering
# Convert categorical columns to category type (numeric columns have already been converted)
data = pd.DataFrame(data)
data[categorical_columns] = data[categorical_columns].astype('category')


# In[31]:


data.info()


# In[32]:


# Check for missing values after conversion
missing_values


# In[33]:


# Encode categorical variables
encoded_data = data.copy()
label_encoders = {}
for column in categorical_columns[:-1]:  # Excluding variable 'y'
    le = LabelEncoder()
    encoded_data[column] = le.fit_transform(data[column])
    label_encoders[column] = le


# In[34]:


encoded_data


# In[35]:


# Encode the target variable 'y'
le_y = LabelEncoder()
encoded_data['y'] = le_y.fit_transform(data['y'])


# In[36]:


encoded_data


# In[37]:


# Define features and target variable 'y'
X = encoded_data.drop(columns='y')
y = encoded_data['y']
X, y


# In[38]:


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# In[39]:


# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
X_train, X_train


# In[40]:


# 3. Predictive model building
# Fit a Random Forest Classifier
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)


# In[41]:


# Predictions
y_pred = rf_model.predict(X_test)
y_prob = rf_model.predict_proba(X_test)[:, 1]
y_pred, y_prob


# In[42]:


# 4. Model performance evaluation
# Generate the confusion matrix, classification report, and ROC curve data
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred, target_names=le_y.classes_)

fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

conf_matrix 


# In[43]:


class_report


# In[44]:


roc_auc


# In[45]:


# Creating dataframes for summary statistics, confusion matrix, and classification report
summary_statistics_df = summary_statistics

conf_matrix_df = pd.DataFrame(conf_matrix, 
                              index=["Actual No", "Actual Yes"], 
                              columns=["Predicted No", "Predicted Yes"])

class_report_df = pd.DataFrame(classification_report(y_test, y_pred, target_names=le_y.classes_, output_dict=True)).transpose()


# In[46]:


summary_statistics_df


# In[47]:


conf_matrix_df


# In[48]:


class_report_df


# In[49]:


# Plotting the ROC Curve
plt.figure(figsize=(10, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()


# In[50]:


# Plotting the confusion matrix as a heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_df, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()


# In[ ]:




