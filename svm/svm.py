import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import confusion_matrix, classification_report, f1_score, accuracy_score

sns.set_style("whitegrid")
plt.rcParams.update({'font.size': 11})

print("Loading data...")
train_data = np.load('train_features.npz')
val_data = np.load('val_features.npz')
test_data = np.load('test_features.npz')

X_train, y_train = train_data['X'], train_data['y']
X_val, y_val = val_data['X'], val_data['y']
X_test, y_test = test_data['X'], test_data['y']

with open('class_names.txt', 'r') as f:
    class_names = [line.strip() for line in f.readlines()]

print("\n--- EXPERIMENT 1: Kernel Function Comparison ---")

kernels = ['linear', 'poly', 'rbf']
kernel_scores = []

for k in kernels:
    clf = SVC(kernel=k, C=10, class_weight='balanced', random_state=42)
    scores = cross_val_score(clf, X_train, y_train, cv=3, scoring='f1_macro')
    mean_score = scores.mean()
    kernel_scores.append(mean_score)
    print(f"Kernel: {k:<10} | CV Macro F1: {mean_score:.4f}")

plt.figure(figsize=(8, 5))
sns.barplot(x=kernels, y=kernel_scores, palette='coolwarm')
plt.title('Impact of Kernel Function on Model Performance')
plt.ylabel('CV Macro F1 Score')
plt.ylim(0, 1.0)
for i, v in enumerate(kernel_scores):
    plt.text(i, v + 0.02, f"{v:.3f}", ha='center', fontsize=11, fontweight='bold')
plt.tight_layout()
plt.savefig('exp1_kernel_comparison.png')
plt.show()

best_kernel_idx = np.argmax(kernel_scores)
best_kernel = kernels[best_kernel_idx]
print(f"\nSelected Kernel: {best_kernel.upper()} (Experiments 2 and 3 will proceed with this kernel)")

print(f"\n--- EXPERIMENT 2: Class Imbalance Strategy (Kernel: {best_kernel}) ---")

strategies = [None, 'balanced']
strategy_names = ['Unbalanced', 'Balanced']

results = {'Accuracy': [], 'Macro F1': []}

for w in strategies:
    clf = SVC(kernel=best_kernel, C=10, class_weight=w, random_state=42)
    
    acc_scores = cross_val_score(clf, X_train, y_train, cv=3, scoring='accuracy')
    f1_scores = cross_val_score(clf, X_train, y_train, cv=3, scoring='f1_macro')
    
    results['Accuracy'].append(acc_scores.mean())
    results['Macro F1'].append(f1_scores.mean())
    
    print(f"Strategy: {str(w):<10} | Accuracy: {acc_scores.mean():.4f} | Macro F1: {f1_scores.mean():.4f}")

x = np.arange(len(strategy_names))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 6))
rects1 = ax.bar(x - width/2, results['Accuracy'], width, label='Accuracy', color='gray')
rects2 = ax.bar(x + width/2, results['Macro F1'], width, label='Macro F1', color='green')

ax.set_ylabel('Scores')
ax.set_title('Unbalanced vs Balanced: Accuracy vs F1 Trade-off')
ax.set_xticks(x)
ax.set_xticklabels(strategy_names)
ax.legend()

ax.bar_label(rects1, padding=3, fmt='%.3f')
ax.bar_label(rects2, padding=3, fmt='%.3f')

plt.tight_layout()
plt.savefig('imbalance_tradeoff.png')
plt.show()

print("\n--- EXPERIMENT 3: Hyperparameter Grid Search (C & Gamma) ---")

param_grid = {
    'C': [0.1, 1, 10, 100], 
    'gamma': ['scale', 'auto', 0.01, 0.1],
    'kernel': [best_kernel],
    'class_weight': ['balanced']
}

grid = GridSearchCV(SVC(random_state=42), param_grid, cv=3, scoring='f1_macro', return_train_score=True, n_jobs=-1)
grid.fit(X_train, y_train)

print(f"Best Parameters: {grid.best_params_}")
print(f"Best CV Score (Macro F1): {grid.best_score_:.4f}")

results_df = pd.DataFrame(grid.cv_results_)
heatmap_data = results_df.pivot(index='param_C', columns='param_gamma', values='mean_test_score')

plt.figure(figsize=(10, 8))
sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='viridis', linewidths=.5)
plt.title(f'Hyperparameter Tuning Results (Macro F1)\nKernel: {best_kernel.upper()}')
plt.ylabel('C Parameter')
plt.xlabel('Gamma Parameter')
plt.yticks(rotation=0) 
plt.tight_layout()
plt.savefig('hyperparameter_heatmap.png')
plt.show()

final_model = grid.best_estimator_

y_pred = final_model.predict(X_test)
test_acc = accuracy_score(y_test, y_pred)
test_f1 = f1_score(y_test, y_pred, average='macro')

print("-" * 30)
print(f"Final Test Accuracy: {test_acc:.4f}")
print(f"Final Test Macro F1: {test_f1:.4f}")
print("-" * 30)

conf_mat = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(20, 18))
sns.heatmap(conf_mat, annot=False, cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.title(f'Final SVM Confusion Matrix\nAccuracy: {test_acc:.3f} | Macro F1: {test_f1:.3f}')
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig('final_confusion_matrix.png')
plt.show()

report_dict = classification_report(y_test, y_pred, target_names=class_names, output_dict=True)
df_report = pd.DataFrame(report_dict).transpose().drop(['accuracy', 'macro avg', 'weighted avg'])
df_report['class_name'] = df_report.index
df_sorted = df_report.sort_values(by='f1-score', ascending=False)

plt.figure(figsize=(10, 8))
sns.barplot(x='f1-score', y='class_name', data=df_sorted.head(15), palette='viridis')
plt.title('Top-15 Classes by F1 Score (Final SVM)')
plt.xlabel('F1 Score')
plt.xlim(0, 1.0)
plt.tight_layout()
plt.savefig('final_top15_f1.png')
plt.show()

plt.figure(figsize=(10, 8))
sns.barplot(x='f1-score', y='class_name', data=df_sorted.tail(15), palette='magma')
plt.title('Bottom-15 Classes by F1 Score (Final SVM)')
plt.xlabel('F1 Score')
plt.xlim(0, 1.0)
plt.tight_layout()
plt.savefig('final_bottom15_f1.png')
plt.show()

print("\n--- Top Confused Classes ---")
np.fill_diagonal(conf_mat, 0)
sorted_indices = np.argsort(conf_mat.flatten())[::-1]

print(f"{'Count':<6} {'True Class':<35} {'Predicted Class':<35}")
print("-" * 80)
for i in range(15):
    idx = sorted_indices[i]
    row, col = divmod(idx, conf_mat.shape[0])
    if conf_mat[row, col] == 0: break
    print(f"{conf_mat[row, col]:<6} {class_names[row]:<35} -> {class_names[col]:<35}")

print("\nAll analyses completed and visualizations saved.")