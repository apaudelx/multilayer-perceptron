import random
import numpy as np
import matplotlib.pyplot as plt
from mlp import Layer, MultilayerPerceptron, CrossEntropy, Relu, Softmax
from mnist_dataloader import MnistDataloader

def show_images(images, title_texts):
    cols = 5
    rows = int(len(images) / cols) + 1
    plt.figure(figsize=(30, 20))
    index = 1
    for image, title_text in zip(images, title_texts):
        plt.subplot(rows, cols, index)
        plt.imshow(image, cmap=plt.cm.gray)
        if title_text != '':
            plt.title(title_text, fontsize=15)
        index += 1
    plt.show()

def main():
    dataloader = MnistDataloader(
        training_images_filepath="data/mnist/train-images.idx3-ubyte",
        training_labels_filepath="data/mnist/train-labels.idx1-ubyte",
        test_images_filepath="data/mnist/t10k-images.idx3-ubyte",
        test_labels_filepath="data/mnist/t10k-labels.idx1-ubyte"
    )
    (train_x, train_y), (test_x, test_y) = dataloader.load_data()
    
    train_x = train_x.astype(np.float32) / 255.0
    test_x = test_x.astype(np.float32) / 255.0
    train_x = train_x.reshape(train_x.shape[0], -1)
    test_x = test_x.reshape(test_x.shape[0], -1)
    
    num_classes = 10
    train_y_onehot = np.eye(num_classes)[train_y]
    test_y_onehot = np.eye(num_classes)[test_y]

    images_2_show = []
    titles_2_show = []
    for i in range(0, 10):
        r = random.randint(0, train_x.shape[0] - 1)
        images_2_show.append(train_x[r].reshape(28, 28))
        titles_2_show.append(f'training image [{r}] = {train_y[r]}')

    for i in range(0, 5):
        r = random.randint(0, test_x.shape[0] - 1)
        images_2_show.append(test_x[r].reshape(28, 28))
        titles_2_show.append(f'test image [{r}] = {test_y[r]}')

    input_size = train_x.shape[1]  # 784 for 28x28 images
    hidden_size = 128
    output_size = num_classes  # 10 classes for MNIST digits

    layer1 = Layer(input_size, hidden_size, Relu(), dropout_rate=0.1)
    layer2 = Layer(128, 64, Relu(), dropout_rate=0.2)
    output_l = Layer(64, output_size, Softmax(), dropout_rate=0.0)
    mlp_model = MultilayerPerceptron([layer1, layer2, output_l])
    
    loss_func = CrossEntropy()

    learning_rate = 1e-3
    batch_size = 64
    epochs = 20

    training_losses, validation_losses = mlp_model.train(
        train_x, train_y_onehot, test_x, test_y_onehot,
        loss_func, learning_rate=learning_rate,
        batch_size=batch_size, epochs=epochs
    )

    print("Training losses:", training_losses)
    print("Validation losses:", validation_losses)
    
    mlp_model.set_training_mode(False)
    
    test_output = mlp_model.forward(test_x)
    predictions = np.argmax(test_output, axis=1)
    true_labels = np.argmax(test_y_onehot, axis=1)
    accuracy = np.mean(predictions == true_labels)
    print("Test Accuracy:", accuracy)
    
    plt.plot(training_losses, label="Training Loss")
    plt.plot(validation_losses, label="Validation Loss")
    plt.legend()
    plt.title("Training and Validation Loss")
    plt.show()
    
    selected_samples = {}
    for idx, pred in enumerate(predictions):
        if pred not in selected_samples:
            selected_samples[pred] = idx
        if len(selected_samples) == 10:
            break

    selected_indices = [selected_samples[i] for i in range(10)]

    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    axes = axes.flatten()
    for ax, idx in zip(axes, selected_indices):
        img = test_x[idx].reshape(28, 28)
        true_label = true_labels[idx]
        pred_label = predictions[idx]
        ax.imshow(img, cmap='gray')
        ax.set_title(f"True: {true_label}\nPred: {pred_label}", fontsize=12)
        ax.axis('off')
    plt.suptitle("One Sample per Class from Test Set", fontsize=16)
    plt.tight_layout()
    plt.show()
    
if __name__ == '__main__':
    main()