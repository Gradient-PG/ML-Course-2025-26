import os
import shutil
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor, Normalize, Compose

from torch.utils.tensorboard import SummaryWriter

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from skorch import NeuralNetClassifier

import optuna
from optuna.integration import TensorBoardCallback

def load_and_preprocess_data(subset_size=5000):
    print(f"\nLoading MNIST data (subset of {subset_size} training samples)...")

    transform = Compose([
        ToTensor(),
        Normalize((0.1307,), (0.3081,)) # MNIST-specific mean/std
    ])

    # Load full datasets
    train_data = MNIST(root="data", train=True, download=True, transform=transform)
    test_data = MNIST(root="data", train=False, download=True, transform=transform)

    # Create a small subset for fast demo
    indices = np.random.choice(len(train_data), subset_size, replace=False)
    train_subset = Subset(train_data, indices)

    # For Grid/Random Search, we need raw X, y
    # Skorch is smart, but this is the most reliable way.
    # We stack the subset into tensors.
    print("Preparing tensors for Skorch (Grid/Random Search)...")
    x_train_tensor = torch.stack([train_subset[i][0] for i in range(len(train_subset))])
    # Labels must be a LongTensor for CrossEntropyLoss
    y_train_tensor = torch.tensor([train_subset[i][1] for i in range(len(train_subset))], dtype=torch.long)

    # For Optuna, DataLoader is much better
    print("Preparing DataLoaders for Optuna (Manual Loop)...")
    train_loader = DataLoader(train_subset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=1024, shuffle=False)

    print("Data loaded and preprocessed.")
    return (x_train_tensor, y_train_tensor), (train_loader, test_loader)

class PytorchCNN(nn.Module):
    def __init__(self, dropout_rate=0.2, num_dense_units=128):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1_in_features = 32 * 14 * 14

        self.fc1 = nn.Linear(self.fc1_in_features, num_dense_units)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(num_dense_units, 10)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = torch.flatten(x, 1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

def run_grid_search(x_train, y_train):
    print("\n--- Starting Section 1: Grid Search ---")

    # Skorch wrapper for PyTorch model
    net = NeuralNetClassifier(
        module=PytorchCNN,
        criterion=nn.CrossEntropyLoss,
        optimizer=optim.Adam,
        max_epochs=3,
        batch_size=32,
        verbose=0,
        lr=0.001,
        module__dropout_rate=0.2,
        module__num_dense_units=128,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )

    param_grid = {
        'lr': [0.0005, 0.001, 0.01],
        'module__num_dense_units': [64, 128, 256],
    }

    print(f"Running GridSearchCV with param grid: {param_grid}")

    grid = GridSearchCV(
        estimator=net,
        param_grid=param_grid,
        cv=2,
        n_jobs=-1,
        verbose=3,
        scoring='accuracy'
    )

    grid_result = grid.fit(x_train, y_train)

    print("\nGrid Search Complete.")
    print(f"Best Score (accuracy): {grid_result.best_score_:.4f}")
    print(f"Best Params: {grid_result.best_params_}")

    try:
        results = grid_result.cv_results_
        scores = results['mean_test_score'].reshape(
            len(param_grid['lr']),
            len(param_grid['module__num_dense_units'])
        )

        plt.figure(figsize=(8, 6))
        plt.imshow(scores, interpolation='nearest', cmap=plt.cm.viridis)
        plt.xlabel('num_dense_units')
        plt.ylabel('learning_rate')
        plt.colorbar(label='Mean Test Accuracy')
        plt.xticks(
            np.arange(len(param_grid['module__num_dense_units'])),
            param_grid['module__num_dense_units']
        )
        plt.yticks(
            np.arange(len(param_grid['lr'])),
            param_grid['lr']
        )
        plt.title('Grid Search Accuracy Heatmap')

        for i in range(len(param_grid['lr'])):
            for j in range(len(param_grid['module__num_dense_units'])):
                plt.text(j, i, f"{scores[i, j]:.3f}", ha='center', va='center', color='white' if scores[i, j] < 0.9 else 'black')

        save_path = "grid_search_heatmap.png"
        plt.savefig(save_path)
        print(f"\nSaved Grid Search heatmap to: {os.path.abspath(save_path)}")

    except Exception as e:
        print(f"\nCould not generate heatmap. Error: {e}")
        print("Note: Heatmap visualization only works for 2D param grids.")


def run_random_search(x_train, y_train):
    print("\n--- Starting Section 2: Random Search ---")

    net = NeuralNetClassifier(
        module=PytorchCNN,
        criterion=nn.CrossEntropyLoss,
        optimizer=optim.Adam,
        max_epochs=3,
        batch_size=32,
        verbose=0,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )

    param_dist = {
        'lr': [0.0001, 0.001, 0.01],
        'module__dropout_rate': [0.1, 0.2, 0.3, 0.4, 0.5],
        'module__num_dense_units': [32, 64, 128, 256]
    }

    print(f"Running RandomizedSearchCV with 10 trials...")

    random_search = RandomizedSearchCV(
        estimator=net,
        param_distributions=param_dist,
        n_iter=10,
        cv=2,
        n_jobs=-1,
        verbose=1,
        scoring='accuracy'
    )

    random_result = random_search.fit(x_train, y_train)

    print("\nRandom Search Complete.")
    print(f"Best Score (accuracy): {random_result.best_score_:.4f}")
    print(f"Best Params: {random_result.best_params_}")

    try:
        results = pd.DataFrame(random_result.cv_results_)
        results = results.sort_values(by='mean_test_score', ascending=False)

        print("\nRandom Search Results Table:")
        param_cols = [
            'param_lr',
            'param_module__dropout_rate',
            'param_module__num_dense_units'
        ]
        cols_to_show = ['mean_test_score'] + param_cols
        cols_to_show = [col for col in cols_to_show if col in results.columns]
        print(results[cols_to_show].to_string(index=False))

        fig, axes = plt.subplots(1, 3, figsize=(20, 6), sharey=True)
        fig.suptitle('Random Search: Individual Parameter vs. Accuracy', fontsize=16) # Removed y=1.02

        sc = axes[0].scatter(
            results['param_lr'],
            results['mean_test_score'],
            c=results['mean_test_score'],
            cmap='viridis',
            alpha=0.7
        )
        axes[0].set_xlabel('Learning Rate (log scale)')
        axes[0].set_ylabel('Mean Test Accuracy')
        axes[0].set_xscale('log')
        axes[0].grid(True, which="both", ls="--", alpha=0.5)

        axes[1].scatter(
            results['param_module__dropout_rate'],
            results['mean_test_score'],
            c=results['mean_test_score'],
            cmap='viridis',
            alpha=0.7
        )
        axes[1].set_xlabel('Dropout Rate')
        axes[1].grid(True, which="both", ls="--", alpha=0.5)

        axes[2].scatter(
            results['param_module__num_dense_units'],
            results['mean_test_score'],
            c=results['mean_test_score'],
            cmap='viridis',
            alpha=0.7
        )
        axes[2].set_xlabel('Num Dense Units')
        axes[2].grid(True, which="both", ls="--", alpha=0.5)

        plt.tight_layout(rect=[0, 0.03, 1, 0.96])

        fig.colorbar(sc, ax=axes.ravel().tolist(), label='Mean Test Accuracy')

        save_path_scatter = "random_search_scatter_plots.png"
        plt.savefig(save_path_scatter)
        print(f"Saved Random Search scatter plots to: {os.path.abspath(save_path_scatter)}")

    except Exception as e:
        print(f"\nCould not generate plots/table. Error: {e}")

def run_optuna_search(train_loader, test_loader):
    print("\n--- Starting Section 3: Optuna + TensorBoard ---")

    log_dir_hparams = "logs/optuna_hparams"
    log_dir_trials = "logs/optuna_trials"
    if os.path.exists("logs"):
        shutil.rmtree("logs")
    print(f"Cleaned up old 'logs' directory.")

    log_dir_abs = os.path.abspath('logs')
    print("\n--- TENSORBOARD (for Optuna) ---")
    print(f"\nTo view results LIVE, start TensorBoard in a NEW terminal:")
    print("---------------------------------------------------------")
    print(f" tensorboard --logdir={log_dir_abs} --port 6006 ")
    print("---------------------------------------------------------")
    print("Then open http://localhost:6006/ in your browser.")

    print("\n-----------------------------------------------\n")
    print("Starting Optuna study (25 trials)...")
    print("Watch TensorBoard as the trials complete!\n")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    def objective(trial: optuna.trial.Trial):
        # 1. Suggest hyperparameters
        learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
        dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.5)
        num_dense_units = trial.suggest_categorical("num_dense_units", [64, 128, 256])

        # 2. Create model, optimizer, criterion
        model = PytorchCNN(dropout_rate, num_dense_units).to(device)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()

        # 3. Create a unique TensorBoard SummaryWriter for this trial
        trial_log_dir = os.path.join(log_dir_trials, f"trial_{trial.number}")
        writer = SummaryWriter(log_dir=trial_log_dir)

        print(f"\nTrial {trial.number}: lr={learning_rate:.6f}, dropout={dropout_rate:.2f}, units={num_dense_units}")

        n_epochs = 5
        final_val_accuracy = 0.0

        for epoch in range(n_epochs):
            model.train()

            train_loss = 0.0
            train_correct = 0
            train_total = 0

            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(device), target.to(device)

                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()

                train_loss += loss.item() * data.size(0)
                pred = output.argmax(dim=1, keepdim=True)
                train_correct += pred.eq(target.view_as(pred)).sum().item()
                train_total += data.size(0)

            avg_train_loss = train_loss / train_total
            avg_train_acc = train_correct / train_total

            # Validation loop
            model.eval()
            val_loss = 0
            correct = 0
            with torch.no_grad():
                for data, target in test_loader:
                    data, target = data.to(device), target.to(device)
                    output = model(data)
                    val_loss += criterion(output, target).item() * data.size(0)
                    pred = output.argmax(dim=1, keepdim=True)
                    correct += pred.eq(target.view_as(pred)).sum().item()

            val_loss /= len(test_loader.dataset)
            val_accuracy = correct / len(test_loader.dataset)
            final_val_accuracy = val_accuracy # Store last accuracy

            writer.add_scalar("Loss/train_epoch", avg_train_loss, epoch)
            writer.add_scalar("Accuracy/train_epoch", avg_train_acc, epoch)
            writer.add_scalar("Loss/validation", val_loss, epoch)
            writer.add_scalar("Accuracy/validation", val_accuracy, epoch)

            print(f"  Epoch {epoch}: Train Acc: {avg_train_acc:.4f} | Val. Acc: {val_accuracy:.4f}")

            trial.report(val_accuracy, epoch)
            if trial.should_prune():
                print(f"  Trial {trial.number} pruned at epoch {epoch}.")
                writer.close()
                raise optuna.exceptions.TrialPruned()

        writer.close()

        return final_val_accuracy

    tensorboard_hparam_callback = TensorBoardCallback(
        dirname=log_dir_hparams,
        metric_name="val_accuracy"
    )

    study = optuna.create_study(direction="maximize", pruner=optuna.pruners.MedianPruner())

    study.optimize(
        objective,
        n_trials=25,
        callbacks=[tensorboard_hparam_callback]
    )

    print("\nOptuna Study Complete.")
    print(f"Best Trial Number: {study.best_trial.number}")
    print(f"Best Score (val_accuracy): {study.best_value:.4f}")
    print("Best Params:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")

    print("\n--- End of Section 3 ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hyperparameter Tuning Lecture Demo (PyTorch)")
    parser.add_argument(
        "--run",
        choices=["grid", "random", "optuna", "all"],
        default="optuna",
        help="Which tuning method to run."
    )
    parser.add_argument(
        "--subset_size",
        type=int,
        default=5000,
        help="Number of training samples to use (for speed)."
    )
    args = parser.parse_args()

    (x_train, y_train), (train_loader, test_loader) = load_and_preprocess_data(subset_size=args.subset_size)

    if args.run == "grid" or args.run == "all":
        run_grid_search(x_train, y_train)

    if args.run == "random" or args.run == "all":
        run_random_search(x_train, y_train)

    if args.run == "optuna" or args.run == "all":
        run_optuna_search(train_loader, test_loader)

        print("\n--- Review your Optuna results in TensorBoard ---")
        print("Go to http://localhost:6006/")
        print("If you haven't, run this in a new terminal:")
        print(f" tensorboard --logdir={os.path.abspath('logs')}")
        print("\nWhat to show in TensorBoard:")
        print("1. 'TIME SERIES' tab: See 'Accuracy/train_epoch' vs 'Accuracy/validation' to spot overfitting!")
        print("2. 'HPARAMS' tab: This is the most important part!")
        print("   - 'TABLE VIEW': See all trials, their params, and their final accuracy.")
        print("   - 'PARALLEL COORDINATES VIEW': See how parameter values relate to the final accuracy.")