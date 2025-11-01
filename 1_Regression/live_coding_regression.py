import csv
import math
import matplotlib.pyplot as plt
import numpy as np

def load_salary_data(csv_path):
    xs = []
    ys = []
    with open(csv_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                x = float(row['YearsExperience'])
                y = float(row['Salary'])
            except Exception as e:
                continue
            xs.append(x)
            ys.append(y)
    return xs, ys

def mse(xs, ys, a, b):
    error = 1
    return error

def gradient_descent(xs, ys, lr=0.001, epochs=100):
    a, b = np.random.rand(), np.random.rand()
    n = len(xs)
    history = []

    return a, b, history

def main():
    csv_path = "Salary_dataset.csv"
    xs, ys = load_salary_data(csv_path)

    # show data
    plt.scatter(xs, ys, label="Dane")
    plt.xlabel("YearsExperience")
    plt.ylabel("Salary")
    plt.title("Data: Experience vs Salary")
    plt.show()

    # train model
    a, b, history = gradient_descent(xs, ys)

    print(f"\nTrained model: y = {a:.2f} * x + {b:.2f}")

    # visualize results
    y_pred = [a * x + b for x in xs]
    plt.scatter(xs, ys, label="Data")
    plt.plot(xs, y_pred, color='red', label="Model")
    plt.legend()
    plt.title("Linear regression trained with Gradient Descent")
    plt.show()

    # plot training history
    plt.plot(history)
    plt.xlabel("Epochs (x10)")
    plt.ylabel("MSE")
    plt.title("Loss over time")
    plt.show()

if __name__ == "__main__":
    main()
