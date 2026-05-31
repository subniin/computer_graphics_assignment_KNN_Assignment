import pickle
import os
import tarfile
import numpy as np
import matplotlib.pyplot as plt


def load_cifar10(root):
    tar_path = os.path.join(root, "cifar-10-python.tar.gz")
    batch_dir = os.path.join(root, "cifar-10-batches-py")
    if not os.path.exists(batch_dir):
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(root)

    def read_batch(path):
        with open(path, "rb") as f:
            d = pickle.load(f, encoding="bytes")
        return d[b"data"].astype(np.float32) / 255.0, np.array(d[b"labels"])

    xs, ys = zip(*[read_batch(os.path.join(batch_dir, f"data_batch_{i}")) for i in range(1, 6)])
    X_train, y_train = np.concatenate(xs), np.concatenate(ys)
    X_test, y_test = read_batch(os.path.join(batch_dir, "test_batch"))
    return X_train, y_train, X_test, y_test


class LinearClassifier:
    def __init__(self, input_dim, num_classes, lr=0.01, reg=1e-3):
        self.W = np.random.randn(input_dim, num_classes) * 0.01
        self.b = np.zeros(num_classes)
        self.lr = lr
        self.reg = reg

    def _softmax(self, scores):
        scores -= scores.max(axis=1, keepdims=True)  # numerical stability
        e = np.exp(scores)
        return e / e.sum(axis=1, keepdims=True)

    def _loss_and_grad(self, X, y):
        N = len(y)
        probs = self._softmax(X @ self.W + self.b)

        loss = -np.log(probs[np.arange(N), y] + 1e-8).mean()
        loss += 0.5 * self.reg * np.sum(self.W ** 2)

        dscores = probs
        dscores[np.arange(N), y] -= 1
        dscores /= N

        dW = X.T @ dscores + self.reg * self.W
        db = dscores.sum(axis=0)
        return loss, dW, db

    def train(self, X, y, epochs=50, batch_size=512):
        N = len(y)
        loss_history = []

        for epoch in range(epochs):
            idx = np.random.permutation(N)
            total_loss, n_batches = 0, 0

            for i in range(0, N, batch_size):
                b = idx[i:i + batch_size]
                loss, dW, db = self._loss_and_grad(X[b], y[b])
                self.W -= self.lr * dW
                self.b -= self.lr * db
                total_loss += loss
                n_batches += 1

            avg_loss = total_loss / n_batches
            loss_history.append(avg_loss)

            if (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch+1:3d}/{epochs} | Loss: {avg_loss:.4f} | Train Acc: {self.accuracy(X, y):.4f}")

        return loss_history

    def predict(self, X):
        return np.argmax(X @ self.W + self.b, axis=1)

    def accuracy(self, X, y):
        return (self.predict(X) == y).mean()


def make_confusion_matrix(y_true, y_pred, num_classes):
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t][p] += 1
    return cm


def main():
    CLASSES = ["airplane", "automobile", "bird", "cat", "deer",
               "dog", "frog", "horse", "ship", "truck"]

    root = os.path.dirname(os.path.abspath(__file__))
    print("Loading CIFAR-10 ...")
    X_train, y_train, X_test, y_test = load_cifar10(root)
    print(f"  Train={X_train.shape}, Test={X_test.shape}\n")

    # Normalize
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0) + 1e-8
    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std

    print("Training ...")
    clf = LinearClassifier(input_dim=3072, num_classes=10, lr=0.01, reg=1e-3)
    loss_history = clf.train(X_train, y_train, epochs=50, batch_size=512)

    test_acc = clf.accuracy(X_test, y_test)
    print(f"\n  Test Accuracy: {test_acc:.4f}")

    cm = make_confusion_matrix(y_test, clf.predict(X_test), num_classes=10)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(loss_history, color="steelblue")
    axes[0].set_title("Training Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True, alpha=0.3)

    im = axes[1].imshow(cm, cmap="Blues")
    axes[1].set_xticks(range(len(CLASSES)))
    axes[1].set_yticks(range(len(CLASSES)))
    axes[1].set_xticklabels(CLASSES, rotation=45, ha="right")
    axes[1].set_yticklabels(CLASSES)
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("True")
    axes[1].set_title("Confusion Matrix")
    threshold = cm.max() * 0.5
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            axes[1].text(j, i, str(cm[i, j]), ha="center", va="center",
                         color="white" if cm[i, j] > threshold else "black", fontsize=7)
    plt.colorbar(im, ax=axes[1])
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
