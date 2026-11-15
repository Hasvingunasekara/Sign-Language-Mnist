# Sign Language MNIST Classification

A convolutional neural network (CNN) project for classifying hand-sign
images from the Sign Language MNIST dataset. The notebook walks through
loading and exploring the image data, preprocessing it, splitting it
into training/validation/test sets, training a CNN with early stopping,
and inspecting predictions.

## Project contents

-   `Sign MNIST.ipynb` --- Jupyter notebook containing the complete
    workflow.
-   `archive (5)/sign_mnist_train.csv` --- training data CSV expected by
    the notebook.
-   `archive (5)/sign_mnist_test.csv` --- test data CSV loaded by the
    notebook.
-   `archive (5)/amer_sign2.png` --- dataset illustration displayed in
    the notebook.

> Place the dataset files in a folder named `archive (5)` beside the
> notebook, or update the paths in the data-loading cells to match your
> local folder structure.

## Dataset

The notebook expects the Sign Language MNIST CSV format:

-   Each row contains a `label` column and pixel values for a 28 × 28
    grayscale image.
-   The notebook separates the `label` column from the pixel data.
-   Pixel values are reshaped to `(28, 28, 1)` and normalized by
    dividing by `255.0`.
-   The model predicts one of 25 classes using a softmax output layer.

The notebook displays numeric class labels; it does not include a
numeric-label-to-letter mapping.

## Workflow

1.  **Import libraries** --- NumPy, pandas, scikit-learn, Matplotlib,
    and TensorFlow/Keras.
2.  **Explore the data** --- display the dataset illustration and
    inspect the training data with `head()`, `describe()`, and its
    columns.
3.  **Preprocess images** --- separate features and labels, reshape
    images to 28 × 28 grayscale arrays, and normalize pixel values.
4.  **Visualize samples** --- display randomly selected hand-sign images
    and their labels.
5.  **Split the data** --- create training, validation, and test subsets
    using a 60% / 20% / 20% split.
6.  **Build the CNN** --- use convolution, max-pooling, flattening, and
    dense layers.
7.  **Train the model** --- train for up to 10 epochs and use early
    stopping based on validation loss.
8.  **Evaluate and inspect predictions** --- report validation and test
    accuracy, plot training curves, and visualize sample predictions and
    misclassified images.

## Model architecture

The notebook defines a Keras Sequential CNN with:

-   `Conv2D(32, 3 × 3, ReLU)`
-   `MaxPooling2D(2 × 2)`
-   `Conv2D(64, 3 × 3, ReLU)`
-   `MaxPooling2D(2 × 2)`
-   `Conv2D(64, 3 × 3, ReLU)`
-   `Flatten`
-   `Dense(64, ReLU)`
-   `Dense(25, Softmax)`

**Compilation settings**

-   Optimizer: Adam
-   Loss: sparse categorical cross-entropy
-   Metric: accuracy
-   Early stopping: monitors `val_loss`, with a patience of 3 epochs and
    restores the best weights.

## Requirements

Install the main dependencies with:

``` bash
pip install numpy pandas scikit-learn matplotlib tensorflow jupyter
```

## How to run

1.  Clone or download this repository.

2.  Put the dataset files in the expected `archive (5)/` directory.

3.  Install the dependencies listed above.

4.  Start Jupyter:

    ``` bash
    jupyter notebook
    ```

5.  Open `Sign MNIST.ipynb` and run the cells from top to bottom.

## Notes

-   The notebook loads both `sign_mnist_train.csv` and
    `sign_mnist_test.csv`, but the model evaluation and prediction cells
    use a train/validation/test split made from the rows in
    `sign_mnist_train.csv`. The separately loaded `sign_mnist_test.csv`
    is not used in the current workflow.
-   The split is performed in two stages with `random_state=42`,
    resulting in 60% training, 20% validation, and 20% test data from
    the training CSV.
-   The printed accuracy after evaluating `X_test` is labeled "Test
    accuracy" in the notebook. The validation accuracy is evaluated
    separately.
-   Dataset paths are relative to the notebook's working directory.
    Adjust them if your folder structure differs.

## Possible future improvements

-   Evaluate the trained model on the separately supplied
    `sign_mnist_test.csv`.
-   Add a mapping from numeric class IDs to sign-language letters for
    more interpretable predictions.
-   Save the trained model and add a reusable prediction function.
-   Add a confusion matrix and per-class precision, recall, and
    F1-score.

## License

Add the license applicable to your project and dataset before
redistributing this repository.
