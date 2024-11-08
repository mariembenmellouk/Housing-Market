import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
import numpy as np
import scipy.stats as stats
from statsmodels.stats.diagnostic import het_breuschpagan

# Load dataset 
data = pd.read_csv(r"C:\Users\merie\OneDrive\Bureau\WGU\D600\Task1-D600\D600 Task1-Submission\D600 Task 1 Dataset 1 Housing Information.csv")

# Calculate descriptive statistics for the selected variables
descriptive_stats = data[['Price', 'CrimeRate', 'SchoolRating']].describe()
print(descriptive_stats)

## Univariate and bivariate visualizations 

# Univariate Visualizations
plt.figure(figsize=(15, 5))

# Histogram for Price
plt.subplot(1, 3, 1)
sns.histplot(data['Price'], bins=30, kde=True)
plt.title('Distribution of Price')
plt.xlabel('Price')

# Histogram for CrimeRate
plt.subplot(1, 3, 2)
sns.histplot(data['CrimeRate'], bins=30, kde=True)
plt.title('Distribution of Crime Rate')
plt.xlabel('Crime Rate')

# Histogram for SchoolRating
plt.subplot(1, 3, 3)
sns.histplot(data['SchoolRating'], bins=30, kde=True)
plt.title('Distribution of School Rating')
plt.xlabel('School Rating')

plt.tight_layout()
plt.show()

# Bivariate Visualizations
plt.figure(figsize=(15, 5))

# Scatter plot for CrimeRate vs Price
plt.subplot(1, 2, 1)
sns.scatterplot(data=data, x='CrimeRate', y='Price')
sns.regplot(data=data, x='CrimeRate', y='Price', scatter=False, color='r')
plt.title('Scatter Plot of Crime Rate vs Price')
plt.xlabel('Crime Rate')
plt.ylabel('Price')

# Scatter plot for SchoolRating vs Price
plt.subplot(1, 2, 2)
sns.scatterplot(data=data, x='SchoolRating', y='Price')
sns.regplot(data=data, x='SchoolRating', y='Price', scatter=False, color='r')
plt.title('Scatter Plot of School Rating vs Price')
plt.xlabel('School Rating')
plt.ylabel('Price')

plt.tight_layout()
plt.show()

# Split the dataset into training and testing sets (80% training, 20% testing)
selected_data = data[['Price', 'CrimeRate', 'SchoolRating']]
train_data, test_data = train_test_split(selected_data, test_size=0.2, random_state=42)

# Save the datasets to CSV files
train_data.to_csv(r"C:\Users\merie\OneDrive\Bureau\WGU\D600\Task1-D600\Training Dataset.csv", index=False)
test_data.to_csv(r"C:\Users\merie\OneDrive\Bureau\WGU\D600\Task1-D600\Testing Dataset.csv", index=False)

print("Datasets created and saved successfully.")

## Use the training dataset to create and perform a regression model
# Load the training dataset
data = pd.read_csv(r"C:\Users\merie\OneDrive\Bureau\WGU\D600\Task1-D600\Training Dataset.csv")

# Define independent variables and a constant 
X = data[['CrimeRate', 'SchoolRating']]
X = sm.add_constant(X)

# Define the dependent variable
y = data['Price']

# Fit the initial model
model = sm.OLS(y, X).fit()

# Assumption Checks
plt.figure(figsize=(8, 6))
plt.scatter(model.fittedvalues, model.resid)
plt.axhline(y=0, color='r', linestyle='--')
plt.title('Residuals vs Fitted Values')
plt.xlabel('Fitted Values')
plt.ylabel('Residuals')
plt.show()

#Normality of Residuals
plt.figure(figsize=(8, 6))
stats.probplot(model.resid, dist="norm", plot=plt)
plt.title('Q-Q Plot')
plt.show()
shapiro_test = stats.shapiro(model.resid)
print(f'Shapiro-Wilk test statistic: {shapiro_test.statistic}, p-value: {shapiro_test.pvalue}')

#Constatnt variance (Breusch-Pagan test)
bp_test = het_breuschpagan(model.resid, model.model.exog)
labels = ['Lagrange multiplier statistic', 'p-value', 'f-value', 'f p-value']
print(dict(zip(labels, bp_test)))

# Summary of the model
print(model.summary())

# Optimize regression model with backward elimination
def backward_elimination(X, y, significance_level=0.05):
    X = X.copy()
    while True:
        model = sm.OLS(y, X).fit()
        p_values = model.pvalues
        
        print(f"Current model p-values:\n{p_values}\n")
        
        max_p_value = p_values.max()
        if max_p_value > significance_level:

        # Remove the variable with the highest p-value    
            excluded_variable = p_values.idxmax()
            print(f"Dropping variable: {excluded_variable} with p-value: {max_p_value}")  
            X = X.drop(columns=[excluded_variable])
        else:
            break
    return model

# Run backward elimination
optimized_model = backward_elimination(X, y)

# Ensure all remaining variables are significant
print("All remaining variables in the optimized model are significant.")

# Summary of the optimized model
print("Optimized Model Summary:")
print(optimized_model.summary())

# Extracting model parameters
params = {
    "Adjusted R²": optimized_model.rsquared_adj,
    "R²": optimized_model.rsquared,
    "F-statistic": optimized_model.fvalue,
    "Probability F-statistic": optimized_model.f_pvalue,
    "Coefficient Estimates": optimized_model.params,
    "P-values": optimized_model.pvalues
}

# Print model parameters
for key, value in params.items():
    print(f"{key}: {value}")

# Prepare the predictors for the optimized model
optimized_features = optimized_model.model.exog 

# Make predictions on the training set using the optimized model
predictions_train = optimized_model.predict(optimized_features)
mse_train = np.mean((y - predictions_train) ** 2)
print(f"Mean Squared Error of the optimized model (training set): {mse_train:.2f}")

# Calculate MSE for the optimized model
mse_optimized = np.mean((y - predictions_train) ** 2)

# Regression equation
intercept = optimized_model.params['const']
school_rating_coef = optimized_model.params.get('SchoolRating', 0)
regression_equation = f"Price = {intercept:.2f} + {school_rating_coef:.2f} * SchoolRating"

# Print the regression equation
print("Regression Equation:")
print(regression_equation)

# Discuss the coefficients
print("\nDiscuss the coefficients:")
print(f"Intercept : {intercept:.2f} - This is the expected price when SchoolRating is zero")
print(f"SchoolRating : {school_rating_coef:.2f} - This indicates that for each increase in SchoolRating, the price is expected to increase by {school_rating_coef:.2f}")

# Discussion of model metrics
print("\nDiscussion of Model Metrics:")
print(f"The R² of the training set is {optimized_model.rsquared:.4f}, indicating that {optimized_model.rsquared * 100:.2f}% of the variance in Price is explained by the model")
print(f"The adjusted R² is {optimized_model.rsquared_adj:.4f}, which considers the number of factors included in the model")

# Prepare the test set predictors
test_X = test_data[['SchoolRating']]  # Use only the relevant feature from test data
test_X = sm.add_constant(test_X)

# Make predictions on the test set using the optimized model
test_predictions = optimized_model.predict(test_X)

# Calculate mean squared error for the test set
mse_test = np.mean((test_data['Price'] - test_predictions) ** 2)

# Print the mean squared errors for training and test sets
print(f"The MSE for the training set is {mse_train:.2f}, while the MSE for the test set is {mse_test:.2f}.")
print(f"The MSE for the optimized model (training set) is {mse_optimized:.2f}")

# Compare the MSE values to evaluate model performance
if mse_train < mse_test:
    print("The training set MSE is lower than the test set MSE, suggesting that the model may be overfitting the training data.")
elif mse_train > mse_test:
    print("The training set MSE is higher than the test set MSE, suggesting that the model may be underfitting the training data.")
else:
    print("The training set MSE is similar to the test set MSE, indicating that the model is generalizing well.")

 # Assuming independent_variables and dependent_variable are defined
independent_variables = ['SchoolRating']  #CrimeRate variable dropped 
dependent_variable = 'Price'

# Predictions on test set
X_test = test_data[independent_variables]
y_test = test_data[dependent_variable]

# Add constant term to the features for the regression model
X_test = sm.add_constant(X_test)  

# Make predictions using the optimized model
test_predictions = optimized_model.predict(X_test)

# Calculate MSE for the test set
mse_test = np.mean((y_test - test_predictions) ** 2)
print(f"The MSE for the test set is {mse_test:.2f}.")
