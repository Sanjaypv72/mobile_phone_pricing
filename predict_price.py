import pandas as pd
import numpy as np
import joblib
import sys


MODEL_PATH = "mobile_price_model.pkl"
SCALER_PATH = "scaler.pkl"
FEATURE_NAMES_PATH = "feature_names.pkl"

PRICE_LABELS = {
    0: "💚 LOW COST",
    1: "💛 MEDIUM COST",
    2: "🟠 HIGH COST",
    3: "🔴 VERY HIGH COST"
}

PRICE_RANGES = {
    0: "Under $150",
    1: "$150 - $350",
    2: "$350 - $650",
    3: "Above $650"
}

def load_artifacts():
    try:
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        feature_names = joblib.load(FEATURE_NAMES_PATH)

        print("Model loaded successfully!\n")
        return model, scaler, feature_names
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        print("Run 'train_model.py' first.")
        sys.exit(1)


def add_engineered_features(df):
    df = df.copy()
    df['total_pixels'] = df['px_height'] * df['px_width']
    df['screen_area'] = df['sc_h'] * df['sc_w']
    df['total_camera_mp'] = df['fc'] + df['pc']
    df['ram_per_weight'] = df['ram'] / (df['mobile_wt'] + 1)
    df['network_score'] = df['three_g'] + df['four_g']

    return df

def predict_price_range(model, scaler, feature_names, phone_data):

    df = pd.DataFrame([phone_data])

    df = add_engineered_features(df)

    for col in feature_names:
        if col not in df.columns:
            df[col] = 0
    X = df[feature_names]
    X_scaled = scaler.transform(X)
    prediction = model.predict(X_scaled)[0]

    if hasattr(model, 'predict_proba'):
        probabilities = model.predict_proba(X_scaled)[0]
    else:
        probabilities = None
    return prediction, probabilities

def interactive_mode(model, scaler, feature_names):
    print("📱 Mobile Phone Price Range Predictor")
    print("=" * 60)
    print("Enter phone specifications:\n")

    phone = {}

    phone['battery_power'] = int(input("Battery Power (mAh, e.g. 3000):"))
    phone['clock_speed'] = float(input("Clock Speed (GHz, e.g. 1.5): "))
    phone['fc'] = int(input("Front Camera (MP, e.g. 8): "))
    phone['int_memory'] = int(input("Internal Memory (GB, e.g. 64): "))
    phone['m_dep'] = float(input("Mobile Depth (cm, e.g. 0.6): "))
    phone['mobile_wt'] = int(input("Weight (grams, e.g. 180): "))
    phone['n_cores'] = int(input("Processor Cores (e.g. 8): "))
    phone['pc'] = int(input("Primary Camera (MP, e.g. 48): "))
    phone['px_height'] = int(input("Pixel Height (e.g. 1920): "))
    phone['px_width'] = int(input("Pixel Width (e.g. 1080): "))
    phone['ram'] = int(input("RAM (MB, e.g. 4096 for 4GB): "))
    phone['sc_h'] = int(input("Screen Height (cm, e.g. 14): "))
    phone['sc_w'] = int(input("Screen Width (cm, e.g. 7): "))
    phone['talk_time'] = int(input("Talk Time (hours, e.g. 15): "))

    print("\nBinary features (enter 1 for Yes, 0 for No):")
    phone['blue'] = int(input("Bluetooth (1/0): "))
    phone['dual_sim'] = int(input("Dual SIM (1/0): "))
    phone['four_g'] = int(input("4G Support (1/0): "))
    phone['three_g'] = int(input("3G Support (1/0): "))
    phone['touch_screen'] = int(input("Touch Screen (1/0): "))
    phone['wifi'] = int(input("WiFi (1/0): "))


    prediction, probabilities = predict_price_range(model, scaler, feature_names, phone)

    print("\n" + "="*60)
    print("Prediction Result")
    print("="*60)

    print(f"\n Price Range: {PRICE_LABELS[prediction]}")
    print(f" Approx. Price: {PRICE_RANGES[prediction]}")

    if probabilities is not None:
        print(f"\n Class Probabilities:")
        for cls, prob in enumerate(probabilities):
            bar = "█" * int(prob * 20)
            print(f"    Class {cls} ({list(PRICE_LABELS.values())[cls][2:]:<16}): {prob * 100:5.1f}%  {bar}")

        print("=" * 60 + "\n")


def predict_from_csv(model, scaler, feature_names, csv_file):
    print(f"Loading phones from '{csv_file}'...'")
    df = pd.read_csv(csv_file)
    original_df = df.copy()
    print(f"Loaded {len(df)} records.")
    df = add_engineered_features(df)
    for col in feature_names:
        if col not in df.columns:
            df[col] = 0
    X = df[feature_names]
    X_scaled = scaler.transform(X)
    prediction = model.predict(X_scaled)

    original_df['predicted_class'] = prediction
    original_df['predicted_label'] = [PRICE_LABELS[p] for p in prediction]
    original_df['approx_price'] = [PRICE_RANGES[p] for p in prediction]

    output_file = 'mobile_predictions_output.csv'
    original_df.to_csv(output_file, index=False)

    print(f"Results saved to '{output_file}'\n")

    print("Distribution of Predicted Price Ranges:")
    for cls in range(4):
        count = (prediction == cls).sum()
        pct = count / len(prediction) * 100
        print(f"  {PRICE_LABELS[cls]}: {count} phones ({pct:.1f}%)")

    # Validate if true labels exist
    if 'price_range' in original_df.columns:
        from sklearn.metrics import classification_report
        print("\n📈 Accuracy vs True Labels:")
        print(classification_report(
            original_df['price_range'],
            prediction,
            target_names=["Low Cost", "Medium Cost", "High Cost", "Very High Cost"]
        ))


if __name__ == '__main__':
    model, scaler, feature_names = load_artifacts()

    if len(sys.argv) > 1:
        if sys.argv[1].endswith('.csv'):
            predict_from_csv(model, scaler, feature_names, sys.argv[1])
        else:
            print("Usage: Python predict_price.py [phone.csv]")
            print("  OR : Python predict_price.py")

    else:
        interactive_mode(model, scaler, feature_names)






















































