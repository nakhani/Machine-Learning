# Perceptron (Perception Neuron)

This repository showcases the **Perceptron Algorithm** applied to both classification and regression tasks. The perceptron is implemented as an object-oriented algorithm and is used to explore linear models in various datasets, including binary classification and weather prediction problems.

---
## Projects

### **1. Surgical Dataset 🏨**
A binary classification task using the perceptron algorithm. 

- **Key Steps**:
  - Train an object-oriented perceptron on the surgical dataset for binary classification.
  - Plot the **accuracy** and **loss** for both training and testing data at each epoch.
  - Calculate and visualize the confusion matrix.
  - Experiment with 5 different activation functions: `sigmoid`, `tanh`, `relu`, `linear`.

---
### **2. Weather Prediction 🌦**
A unique regression problem focused on hourly weather data collected over multiple years.

- **Data Preprocessing**:
  - Convert dates into "day of the year" format (e.g., Ordibehesht 2 → Day 33, Esfand 29 → Day 365).
  - Compute daily average temperatures from 24 hourly temperature records for each day.

- **Linear Model Training**:
  - Train a linear perceptron model on the weather dataset.
  - Implement the following functions:
    - **evaluate**: Calculate loss and accuracy for the perceptron model.
    - **predict**: Predict daily temperatures based on the "day of the year."
  - Plot the **loss** and **accuracy** progression during training.

- **Visualization**:
  - Generate a table of daily average temperatures and their corresponding "day of the year."
  - Visualize the data and model predictions on a chart.

- **Stored Results**:
  - Save trained weights (`weights.npy`) and biases (`bias.npy`) to files for future use.

- **Limitations**:
  - Although the data is non-linear, a linear perceptron is intentionally used to create a baseline. Future comparisons will be made with non-linear models.

---


## Features

### **Custom Perceptron Implementation**
- Object-oriented structure for modularity and flexibility.
- Support for multiple activation functions (`sigmoid`, `tanh`, `relu`, `linear`).

### **Data Visualization**
- Plots for loss and accuracy at each epoch for both training and testing phases.
- Confusion matrix visualization for classification tasks.
- Scatterplots for regression predictions and data.

### **Data Preprocessing**
- Efficient handling of time-based weather data.
- Transformation of dates into "day of the year" format.
- Computation of daily averages from hourly temperature records.

### **Model Training**
- Hyperparameter tuning: Learn and adjust learning rates, epochs, and activation functions.
- Save model weights and biases for reproducibility (`weights.npy`, `bias.npy`).

---

### **Result:**

#### **1.Surgical Dataset 🏨**

- Loss and Accuracy Plots:

  <img src = "Employee's salary/Figure_2.png" width = "400">


- Confusion Matrix:

  <img src = "Employee's salary/Figure_2.png" width = "400">

- Evaluated Table:


#### **2.Weather Prediction 🌦**

- Loss and Accuracy Plots:

  <img src = "Abalone/Figure_1.png" width = "400">

- Evaluated Table: 

---

## How to Run the Code
1. Clone the repository:
   ```sh
   https://github.com/nakhani/Machine-Learning/tree/69e66653b4912ed9ad991a678472959065e248a7/Perceptron
   ```

2. Navigate to the directory:
   ```sh
   Perceptron_2
   ```

3. Install the required packages:
   ```sh
   pip install -r requirements.txt
   ```

4. Run the assignments:

   ```sh
    jupyter notebook surgical.ipynb # For predict wether a person has a disorder sign with perceptron 
    jupyter notebook weather.ipynb # For predict the temperature in a day  with perceptron 
   ```

## Dependencies
- Python (Pandas, NumPy, Matplotlib, SciPy, Scikit-learn)
- Jupyter Notebook
