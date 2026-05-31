import tarfile
import pickle
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

# 1단계: 데이터 로드 및 전처리
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

    # 서브셋 추출 및 타입 변환 (연산 오버플로우 방지를 위해 float32 사용)
    X_train = X_train_full[:num_training].astype(np.float32)
    y_train = y_train_full[:num_training]
    X_test = X_test[:num_test].astype(np.float32)
    y_test = y_test[:num_test]

    return X_train, y_train, X_test, y_test

# 2단계: K-Nearest Neighbors 구현 (From Scratch)
class KNearestNeighbor:
    def __init__(self):
        pass

    def train(self, X, y):
        self.X_train = X
        self.y_train = y

    def compute_distances_l1(self, X):
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        # 메모리 효율을 위해 테스트 데이터 기준으로 순회
        for i in range(num_test):
            dists[i, :] = np.sum(np.abs(self.X_train - X[i, :]), axis=1)
        return dists

    def compute_distances_l2(self, X):
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        for i in range(num_test):
            dists[i, :] = np.sqrt(np.sum(np.square(self.X_train - X[i, :]), axis=1))
        return dists

    def predict_labels(self, dists, k=1):
        num_test = dists.shape[0]
        y_pred = np.zeros(num_test, dtype=int)
        for i in range(num_test):
            closest_y = self.y_train[np.argsort(dists[i, :])[:k]]
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
            y_val_pred_l1 = knn.predict_labels(dists_l1, k)
            y_val_pred_l2 = knn.predict_labels(dists_l2, k)
            
            acc_l1 = np.mean(y_val_pred_l1 == y_val)
            acc_l2 = np.mean(y_val_pred_l2 == y_val)
            
            cv_accuracies_l1[k].append(acc_l1)
            cv_accuracies_l2[k].append(acc_l2)
        print(f"Fold {i+1}/5 ")

    mean_acc_l1 = [np.mean(cv_accuracies_l1[k]) for k in k_choices]
    mean_acc_l2 = [np.mean(cv_accuracies_l2[k]) for k in k_choices]
    
    for idx, k in enumerate(k_choices):
        print(f"K={k} -> L1 mean accuracy: {mean_acc_l1[idx]:.4f}, L2 mean accuracy: {mean_acc_l2[idx]:.4f}")
    
    best_k = k_choices[np.argmax(mean_acc_l1)]
    print(f"\nBest hyperparameter: Distance=L1, K={best_k}")

    print("\ntest by best hyperparameter")
    final_knn = KNearestNeighbor()
    final_knn.train(X_train, y_train)
    dists_test = final_knn.compute_distances_l1(X_test)
    y_test_pred = final_knn.predict_labels(dists_test, k=best_k)
    
    test_accuracy = np.mean(y_test_pred == y_test)
    print(f"Last test Accuracy: {test_accuracy:.4f}")
    
    print_confusion_matrix(y_test, y_test_pred)
    
    plt.figure(figsize=(10, 6))
    plt.plot(k_choices, mean_acc_l1, marker='o', label='L1 Distance')
    plt.plot(k_choices, mean_acc_l2, marker='s', label='L2 Distance')
    plt.title('Cross-validation on K')
    plt.xlabel('K value')
    plt.ylabel('Mean cross-validation accuracy')
    plt.xticks(k_choices)
    plt.legend()
    plt.grid(True)
    plt.show()
