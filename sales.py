import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

# Load data
# Load data
try:
    sales_data = pd.read_csv(r'C:\Users\Sai Sunil\sales.csv')
except FileNotFoundError:
    print("File not found. Please check the file path.")
    exit()
except Exception as e:
    print("An error occurred while reading the CSV file:", e)
    exit()

# Check column names
print("Column Names:", sales_data.columns)

# Check if 'Sales' column exists
if 'Sales' not in sales_data.columns:
    print("The 'Sales' column does not exist in the DataFrame.")
    exit()

# Preprocessing (assuming data preprocessing steps are done)
# Separate features (X) and target variable (y)
X = sales_data.drop('Sales', axis=1)
y = sales_data['Sales']

# Print first few rows of the DataFrame
print(X.head())
print(y.head())

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize Linear Regression model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

# Evaluate the model
mae = mean_absolute_error(y_test, predictions)
print("Mean Absolute Error:", mae)
