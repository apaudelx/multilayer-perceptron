import pandas as pd

def load_auto_mpg_data(filepath):
    column_names = ['mpg', 'cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'model_year', 'origin', 'car_name']
    data = pd.read_csv(filepath, delim_whitespace=True, names=column_names, na_values='?')
    data = data.dropna()  # Drop rows with missing values
    data = data.drop('car_name', axis=1)  # Drop the car_name column

    # Normalize the features
    features = data.drop('mpg', axis=1)
    features = (features - features.mean()) / features.std()

    # Extract the target variable
    target = data['mpg']

    return features.values, target.values