import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score,
    recall_score, classification_report, confusion_matrix
)
import joblib
import warnings

warnings.filterwarnings('ignore')

DATASET_PATH = "mobile_data.csv"
MODEL_SAVE_PATH = "mobile_price_model.pkl"
SCALER_SAVE_PATH = "scaler.pkl"

PRICE_LABELS = {
    0: "Low Cost",
    1: "Medium Cost",
    2: "High Cost",
    3: "Very High Cost"
}

print("=" * 70)
print("📱 Mobile Phone Price Range Prediction - Model Training")
print("=" * 70)


def load_dataset(file_path):
    print("\n📂 Loading dataset...")
    df = pd.read_csv(file_path)

    print(f"✓ Dataset loaded: {len(df)} mobile phones, {len(df.columns)} columns")
    print(f"\nPreview:")
    print(df.head())

    return df


def explore_data(df):
    print(f"\n{'=' * 70}")
    print("📊 Exploratory Data Analysis")
    print("=" * 70)

    print("\n1. Dataset Info:")
    print(f"   Rows: {len(df)}, Columns: {len(df.columns)}")
    print(f"   Total Missing Values: {df.isnull().sum().sum()}")

    print("\n2. Price Range Distribution:")
    counts = df['price_range'].value_counts().sort_index()
    for label, count in counts.items():
        pct = (count / len(df)) * 100
        print(f"   {PRICE_LABELS[label]} (Class {label}): {count} phones ({pct:.1f}%)")

    if counts.nunique() == 1 or counts.std() < 50:
        print("\n   ✅ Dataset is perfectly balanced - No SMOTE needed!")

    print("\n3. Key Feature Statistics:")
    key_features = ['ram', 'battery_power', 'px_height', 'px_width', 'int_memory', 'pc']
    print(df[key_features].describe().round(2))

    print("\n4. Binary Features (0/1 flags):")
    binary_cols = ['blue', 'dual_sim', 'four_g', 'three_g', 'touch_screen', 'wifi']
    for col in binary_cols:
        pct = df[col].mean() * 100
        print(f"   {col}: {pct:.1f}% phones have it")

    print("\n5. Avg RAM per Price Range (higher RAM = higher price?):")
    print(df.groupby('price_range')['ram'].mean().round(1).to_string())

    print(f"\n{'=' * 70}\n")


def engineer_features(df):
    print("🔧 Feature Engineering...")

    df = df.copy()

    df['total_pixels'] = df['px_height'] * df['px_width']
    print("  ✓ total_pixels = px_height × px_width")

    df['screen_area'] = df['sc_h'] * df['sc_w']
    print("  ✓ screen_area = sc_h × sc_w")

    df['total_camera_mp'] = df['fc'] + df['pc']
    print("  ✓ total_camera_mp = fc + pc")

    df['ram_per_weight'] = df['ram'] / (df['mobile_wt'] + 1)
    print("  ✓ ram_per_weight = ram / mobile_wt")

    df['network_score'] = df['three_g'] + df['four_g']
    print("  ✓ network_score = three_g + four_g")

    print(f"\n✓ Feature engineering complete! Total features: {len(df.columns) - 1}")
    print(f"{'=' * 70}\n")

    return df


def split_data(df):
    print("📊 Splitting Data...")

    X = df.drop('price_range', axis=1)
    y = df['price_range']

    feature_names = X.columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print(f"✓ Training samples: {len(X_train)}")
    print(f"✓ Testing samples:  {len(X_test)}")
    print(f"✓ Number of features: {len(feature_names)}")

    print(f"\n  Train class distribution:")
    for cls, cnt in y_train.value_counts().sort_index().items():
        print(f"    Class {cls} ({PRICE_LABELS[cls]}): {cnt} ({cnt / len(y_train) * 100:.1f}%)")

    print(f"\n{'=' * 70}\n")
    return X_train, X_test, y_train, y_test, feature_names


def scale_features(X_train, X_test):
    print("⚖️ Scaling Features...")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("✓ StandardScaler applied (mean=0, std=1)")
    print(f"\n{'=' * 70}\n")

    return X_train_scaled, X_test_scaled, scaler


def train_models(X_train, y_train):
    print("🤖 Training Classification Models...")
    print("=" * 70)

    models = {}

    print("\n1. Training Logistic Regression...")
    lr = LogisticRegression(
        multi_class='multinomial',
        max_iter=1000,
        C=1.0,
        random_state=42
    )
    lr.fit(X_train, y_train)
    models['Logistic Regression'] = lr
    print("   ✓ Done")

    print("\n2. Training K-Nearest Neighbors (k=9)...")
    knn = KNeighborsClassifier(
        n_neighbors=9,
        metric='euclidean'
    )
    knn.fit(X_train, y_train)
    models['KNN'] = knn
    print("   ✓ Done")

    print("\n3. Training SVM (RBF kernel)...")
    svm = SVC(
        kernel='rbf',
        C=10,
        gamma='scale',
        probability=True,
        random_state=42
    )
    svm.fit(X_train, y_train)
    models['SVM'] = svm
    print("   ✓ Done")

    print("\n4. Training Decision Tree...")
    dt = DecisionTreeClassifier(
        max_depth=10,
        min_samples_split=5,
        random_state=42
    )
    dt.fit(X_train, y_train)
    models['Decision Tree'] = dt
    print("   ✓ Done")

    print("\n5. Training Random Forest (100 trees)...")
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=15,
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)
    models['Random Forest'] = rf
    print("   ✓ Done")

    print("\n6. Training Gradient Boosting...")
    gb = GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        random_state=42
    )
    gb.fit(X_train, y_train)
    models['Gradient Boosting'] = gb
    print("   ✓ Done")

    print(f"\n{'=' * 70}\n")
    return models


def evaluate_models(models, X_train, X_test, y_train, y_test):
    print("📈 Model Evaluation")
    print("=" * 70)

    results = []

    for model_name, model in models.items():
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        results.append({
            'Model': model_name,
            'Train Accuracy': accuracy_score(y_train, y_train_pred),
            'Test Accuracy': accuracy_score(y_test, y_test_pred),
            'Macro F1': f1_score(y_test, y_test_pred, average='macro'),
            'Macro Precision': precision_score(y_test, y_test_pred, average='macro'),
            'Macro Recall': recall_score(y_test, y_test_pred, average='macro')
        })

    results_df = pd.DataFrame(results).sort_values('Test Accuracy', ascending=False)

    print("\nModel Performance Comparison:")
    print(results_df.to_string(index=False))

    best_model_name = results_df.iloc[0]['Model']
    best_acc = results_df.iloc[0]['Test Accuracy']

    print(f"\n{'=' * 70}")
    print(f"🏆 Best Model: {best_model_name}")
    print(f"🎯 Test Accuracy: {best_acc * 100:.2f}%")
    print(f"{'=' * 70}\n")

    best_model = models[best_model_name]
    y_pred = best_model.predict(X_test)

    print(f"\nDetailed Report - {best_model_name}:")
    print(classification_report(
        y_test, y_pred,
        target_names=list(PRICE_LABELS.values())
    ))

    return results_df, best_model_name


def visualize_results(models, X_test, y_test, best_model_name, feature_names):
    print("📊 Generating Visualizations...")

    best_model = models[best_model_name]
    y_pred = best_model.predict(X_test)

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    cm = confusion_matrix(y_test, y_pred)
    labels = list(PRICE_LABELS.values())
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=labels, yticklabels=labels,
        ax=axes[0]
    )
    axes[0].set_xlabel('Predicted Label')
    axes[0].set_ylabel('True Label')
    axes[0].set_title(f'Confusion Matrix\n({best_model_name})')
    axes[0].tick_params(axis='x', rotation=20)
    axes[0].tick_params(axis='y', rotation=0)

    model_names = list(models.keys())
    accuracies = [
        accuracy_score(y_test, m.predict(X_test)) * 100
        for m in models.values()
    ]
    colors = ['gold' if n == best_model_name else 'steelblue' for n in model_names]

    bars = axes[1].bar(model_names, accuracies, color=colors, edgecolor='black')
    axes[1].set_ylabel('Test Accuracy (%)')
    axes[1].set_title('Model Accuracy Comparison')
    axes[1].set_ylim(60, 105)
    axes[1].tick_params(axis='x', rotation=20)
    axes[1].grid(True, axis='y', alpha=0.3)
    for bar, acc in zip(bars, accuracies):
        axes[1].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.3,
            f'{acc:.1f}%',
            ha='center', va='bottom', fontsize=9
        )

    if hasattr(best_model, 'feature_importances_'):
        importances = best_model.feature_importances_
        top_n = 12
        top_idx = np.argsort(importances)[-top_n:]

        axes[2].barh(
            [feature_names[i] for i in top_idx],
            importances[top_idx],
            color='steelblue', edgecolor='black'
        )
        axes[2].set_xlabel('Importance Score')
        axes[2].set_title(f'Top {top_n} Feature Importance\n({best_model_name})')
        axes[2].grid(True, alpha=0.3)
    else:
        axes[2].text(0.5, 0.5,
                     'Feature importance\nnot available for\nthis model type.\nSee README for\ntop features.',
                     ha='center', va='center', fontsize=12)
        axes[2].set_title('Feature Importance')

    plt.tight_layout()
    plt.savefig('mobile_price_results.png', dpi=150, bbox_inches='tight')
    print("✓ Visualizations saved as 'mobile_price_results.png'\n")
    plt.close()


def save_artifacts(best_model_name, models, scaler, feature_names):
    print("💾 Saving Model Artifacts...")

    joblib.dump(models[best_model_name], MODEL_SAVE_PATH)
    joblib.dump(scaler, SCALER_SAVE_PATH)
    joblib.dump(feature_names, 'feature_names.pkl')

    print(f"✓ Model saved:        '{MODEL_SAVE_PATH}'")
    print(f"✓ Scaler saved:       '{SCALER_SAVE_PATH}'")
    print(f"✓ Feature names saved: 'feature_names.pkl'")
    print(f"\n{'=' * 70}\n")


if __name__ == "__main__":
    df = load_dataset(DATASET_PATH)

    explore_data(df)

    df = engineer_features(df)

    X_train, X_test, y_train, y_test, feature_names = split_data(df)

    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)

    models = train_models(X_train_scaled, y_train)

    results_df, best_model_name = evaluate_models(
        models, X_train_scaled, X_test_scaled, y_train, y_test
    )

    visualize_results(models, X_test_scaled, y_test, best_model_name, feature_names)

    save_artifacts(best_model_name, models, scaler, feature_names)

    print("🎉 Training Complete!")
    print("Use 'predict_price.py' to predict pricing for new phones.\n")