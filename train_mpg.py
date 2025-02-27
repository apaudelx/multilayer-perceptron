import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from mpg_dataloader import load_auto_mpg_data
import mlp

def main():
    # Load the Auto MPG dataset
    X, y = load_auto_mpg_data('data/auto-mpg/auto-mpg.data')

    # Ensure that X and y are NumPy arrays (in case the loader returns a DataFrame)
    if not isinstance(X, np.ndarray):
        X = np.array(X)
    if not isinstance(y, np.ndarray):
        y = np.array(y)

    # Split data: 70% training, 15% validation, 15% test
    X_train, X_leftover, y_train, y_leftover = train_test_split(
        X, y, test_size=0.3, random_state=42, shuffle=True
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_leftover, y_leftover, test_size=0.5, random_state=42, shuffle=True
    )

    # Compute statistics for X (features)
    X_mean = X_train.mean(axis=0)
    X_std = X_train.std(axis=0)

    # Standardize features
    X_train = (X_train - X_mean) / X_std
    X_val   = (X_val - X_mean) / X_std
    X_test  = (X_test - X_mean) / X_std

    # Compute statistics for y (targets)
    y_mean = y_train.mean()
    y_std = y_train.std()

    # Standardize targets
    y_train = (y_train - y_mean) / y_std
    y_val = (y_val - y_mean) / y_std
    y_test = (y_test - y_mean) / y_std

    # Reshape y to column vectors if needed for the MLP
    if len(y_train.shape) == 1:
        y_train = y_train.reshape(-1, 1)
        y_val = y_val.reshape(-1, 1)
        y_test = y_test.reshape(-1, 1)

    # Define MLP architecture using a deeper network with dropout
    input_size = X_train.shape[1]  # Typically 7 features for Auto MPG
    hidden_layer1 = mlp.Layer(input_size, 128, mlp.Relu(), dropout_rate=0.2)
    hidden_layer2 = mlp.Layer(128, 64, mlp.Relu(), dropout_rate=0.3)
    hidden_layer3 = mlp.Layer(64, 32, mlp.Relu(), dropout_rate=0.2)
    hidden_layer4 = mlp.Layer(32, 16, mlp.Relu(), dropout_rate=0.0)
    output_layer = mlp.Layer(16, 1, mlp.Linear(), dropout_rate=0.0)

    # MLP model
    model = mlp.MultilayerPerceptron([hidden_layer1, hidden_layer2, hidden_layer3, hidden_layer4, output_layer])
    
    # loss function
    loss_func = mlp.SquaredError()

    # training hyperparameters
    learning_rate = 1e-3
    batch_size = 16
    epochs = 200

    # train the model
    train_losses, val_losses = model.train(
        X_train, y_train,
        X_val, y_val,
        loss_func,
        learning_rate=learning_rate,
        batch_size=batch_size,
        epochs=epochs,
        rmsprop=True
    )
    
    # Plot loss curves for training and validation
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    ax1.plot(train_losses, color='b', label='Training')
    ax1.plot(val_losses, color='r', label='Validation')
    ax1.set_title("Loss Curves", size=14)
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Mean Squared Error")
    ax1.legend()
    ax1.grid(True)
    
    model.set_training_mode(False)
    test_predictions = model.forward(X_test)
    test_mse = mean_squared_error(y_test, test_predictions)
    test_r2 = r2_score(y_test, test_predictions)
    
    test_predictions_original = test_predictions * y_std + y_mean
    y_test_original = y_test * y_std + y_mean
    
    ax2.scatter(y_test_original, test_predictions_original, alpha=0.5)
    ax2.plot([y_test_original.min(), y_test_original.max()], 
             [y_test_original.min(), y_test_original.max()], 'r--', lw=2)
    ax2.set_title("Actual vs Predicted MPG", size=14)
    ax2.set_xlabel("Actual MPG")
    ax2.set_ylabel("Predicted MPG")
    ax2.grid(True)

    plt.tight_layout()
    plt.show()
    
    print("\nModel Performance Metrics:")
    print(f"Test MSE: {test_mse:.4f}")
    print(f"Test R² Score: {test_r2:.4f}")

    np.random.seed(42)
    sample_indices = np.random.choice(len(X_test), 10, replace=False)
    samples = {
        "Sample #": [],
        "True MPG": [],
        "Predicted MPG": []
    }
    for idx in sample_indices:
        samples["Sample #"].append(idx)
        samples["True MPG"].append(round(y_test_original[idx, 0], 2))
        samples["Predicted MPG"].append(round(test_predictions_original[idx, 0], 2))

    sample_table = pd.DataFrame(samples)
    print("\nSample Predictions on Test Set:")
    print(sample_table.to_string(index=False))
    
if __name__ == '__main__':
    main()
