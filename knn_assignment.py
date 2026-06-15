import tarfile
import pickle
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

def load_cifar10_subset(tar_path, num_training=5000, num_test=1000):
    x_train_batches, y_train_batches = [], []
    X_test, y_test = None, None

    with tarfile.open(tar_path, 'r:gz') as tar:
        for member in tar.getmembers():
            if 'data_batch' in member.name:
                f = tar.extractfile(member)
                dict_data = pickle.load(f, encoding='bytes')
                x_train_batches.append(dict_data[b'data'])
                y_train_batches.append(dict_data[b'labels'])
            elif 'test_batch' in member.name:
                f = tar.extractfile(member)
                dict_data = pickle.load(f, encoding='bytes')
                X_test, y_test = dict_data[b'data'], np.array(dict_data[b'labels'])

    X_train_full = np.concatenate(x_train_batches)
    y_train_full = np.concatenate(y_train_batches)

    X_train = X_train_full[:num_training].astype(np.float32)
    y_train = y_train_full[:num_training]
    X_test = X_test[:num_test].astype(np.float32)
    y_test = y_test[:num_test]

    return X_train, y_train, X_test, y_test


class KNearestNeighbor:
    def train(self, X, y):
        self.X_train = X
        self.y_train = y

    def compute_distances_l1(self, X):
        dists = np.zeros((X.shape[0], self.X_train.shape[0]))
        for i in range(X.shape[0]):
            dists[i] = np.sum(np.abs(self.X_train - X[i]), axis=1)
        return dists

    def compute_distances_l2(self, X):
        dists = np.zeros((X.shape[0], self.X_train.shape[0]))
        for i in range(X.shape[0]):
            dists[i] = np.sqrt(np.sum(np.square(self.X_train - X[i]), axis=1))
        return dists

    def predict_labels(self, dists, k=1):
        y_pred = np.zeros(dists.shape[0], dtype=int)
        for i in range(dists.shape[0]):
            closest_y = self.y_train[np.argsort(dists[i])[:k]]
            y_pred[i] = Counter(closest_y).most_common(1)[0][0]
        return y_pred


def print_confusion_matrix(y_true, y_pred, num_classes=10):
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    print("\n[Confusion Matrix]")
    print(cm)


if __name__ == "__main__":
    tar_file = "cifar-10-python.tar.gz"
    X_train, y_train, X_test, y_test = load_cifar10_subset(tar_file, 5000, 1000)

    # flatten: 32x32x3 -> 3072
    X_train = X_train.reshape(len(X_train), -1)
    X_test = X_test.reshape(len(X_test), -1)

    num_folds = 5
    k_choices = [1, 3, 5, 7, 9]

    X_train_folds = np.array_split(X_train, num_folds)
    y_train_folds = np.array_split(y_train, num_folds)

    cv_accuracies_l1 = {k: [] for k in k_choices}
    cv_accuracies_l2 = {k: [] for k in k_choices}

    print("\n5-Fold start")
    for i in range(num_folds):
        X_val = X_train_folds[i]
        y_val = y_train_folds[i]
        X_tr = np.concatenate(X_train_folds[:i] + X_train_folds[i+1:])
        y_tr = np.concatenate(y_train_folds[:i] + y_train_folds[i+1:])

        knn = KNearestNeighbor()
        knn.train(X_tr, y_tr)
        dists_l1 = knn.compute_distances_l1(X_val)
        dists_l2 = knn.compute_distances_l2(X_val)

        for k in k_choices:
            cv_accuracies_l1[k].append(np.mean(knn.predict_labels(dists_l1, k) == y_val))
            cv_accuracies_l2[k].append(np.mean(knn.predict_labels(dists_l2, k) == y_val))
        print(f"Fold {i+1}/5")

    mean_acc_l1 = [np.mean(cv_accuracies_l1[k]) for k in k_choices]
    mean_acc_l2 = [np.mean(cv_accuracies_l2[k]) for k in k_choices]

    for k, a1, a2 in zip(k_choices, mean_acc_l1, mean_acc_l2):
        print(f"K={k} -> L1: {a1:.4f}, L2: {a2:.4f}")

    best_k = k_choices[np.argmax(mean_acc_l1)]
    print(f"\nBest hyperparameter: Distance=L1, K={best_k}")

    print("\ntest by best hyperparameter...")
    final_knn = KNearestNeighbor()
    final_knn.train(X_train, y_train)
    dists_test = final_knn.compute_distances_l1(X_test)
    y_test_pred = final_knn.predict_labels(dists_test, k=best_k)

    print(f"Last test Accuracy: {np.mean(y_test_pred == y_test):.4f}")
    print_confusion_matrix(y_test, y_test_pred)

    # plot
    fig, ax = plt.subplots(figsize=(10, 6))

    for k in k_choices:
        ax.scatter([k] * num_folds, cv_accuracies_l1[k], color='steelblue', alpha=0.3, s=25)
        ax.scatter([k] * num_folds, cv_accuracies_l2[k], color='tomato', alpha=0.3, s=25)

    ax.plot(k_choices, mean_acc_l1, marker='o', color='steelblue', linewidth=2, label='L1 Distance')
    ax.plot(k_choices, mean_acc_l2, marker='s', color='tomato', linewidth=2, label='L2 Distance')

    for k, a1, a2 in zip(k_choices, mean_acc_l1, mean_acc_l2):
        ax.annotate(f'{a1:.3f}', xy=(k, a1), xytext=(-20, 6), textcoords='offset points', color='steelblue', fontsize=9)
        ax.annotate(f'{a2:.3f}', xy=(k, a2), xytext=(5, 6), textcoords='offset points', color='tomato', fontsize=9)

    ax.axvline(x=best_k, linestyle='--', color='gray', alpha=0.7, label=f'Best K={best_k} (L1)')
    ax.set_title('5-Fold Cross-Validation Accuracy per K')
    ax.set_xlabel('K value')
    ax.set_ylabel('Mean validation accuracy')
    ax.set_xticks(k_choices)
    ax.legend()
    ax.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.show()
