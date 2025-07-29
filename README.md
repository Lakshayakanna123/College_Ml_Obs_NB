# Naive Bayes Weather Classifier Example

This project demonstrates how to use the **WEKA** machine learning library in Java to train and evaluate a Naive Bayes classifier on a simple weather dataset formatted in ARFF.

## Overview

This program:
- Loads sample weather data encoded as a string in ARFF format.
- Uses WEKA's `NaiveBayes` classifier to build a model predicting whether to play tennis (`play`: yes/no) based on weather attributes (`outlook`, `temperature`, `humidity`, `windy`).
- Evaluates the model using 10-fold cross-validation.
- Prints accuracy and class-level statistics.

## Requirements

- **Java 8 or above**
- **WEKA library (weka.jar)** must be available in the classpath.

## How It Works

1. **ARFF Data**: The weather dataset is embedded as a string in ARFF format.
2. **Loading Data**: Uses `ArffReader` to convert the ARFF string into WEKA `Instances`.
3. **Setting Class Attribute**: The target class (`play`) is set as the last attribute.
4. **Model Training**: A `NaiveBayes` classifier is trained on the data.
5. **Evaluation**: The model is evaluated using 10-fold cross-validation via the `Evaluation` class.
6. **Results**: The program prints:
    - Overall **accuracy**
    - **Precision, recall, F1-score** for each class

## Usage

### Compile

```sh
javac -cp weka.jar nb.java
```

### Run

```sh
java -cp .:weka.jar nb
```

On Windows, replace `:` with `;` in the classpath.

## Notes

- The ARFF data is hard-coded, but you can modify it to test other scenarios.
- The program prints accuracy and detailed classification statistics to the console.
- Other outputs (model details, confusion matrix) can be uncommented if needed.

## Example Output

```
Accuracy: 78.57142857142857%
=== Detailed Accuracy By Class ===

               TP Rate   FP Rate   Precision   Recall  F-Measure   MCC    ROC Area   PRC Area  Class
                 0.833     0.333      0.833     0.833     0.833   0.500     0.833     0.905      yes
                 0.667     0.167      0.800     0.667     0.727   0.500     0.833     0.648       no
```

## File Structure

- `nb.java` — Main program source file.

## Customization

- Change the ARFF data to your own dataset (keep attributes/types consistent).
- Adjust the classifier or evaluation settings as needed for your experiments.

## License

This code is provided for educational purposes and does not include a specific license.

For more information on WEKA and its capabilities, refer to the official WEKA documentation.
