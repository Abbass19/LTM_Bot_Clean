import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats import skew, kurtosis
from sklearn.preprocessing import RobustScaler, MinMaxScaler

plt.rcParams['font.family'] = 'Segoe UI Emoji'

def analyze_distribution(data, feature_names=None, feature_label=None, summary_report=None):
    # Initialize summary report
    if summary_report is None:
        summary_report = []

    # Handle multi-feature input
    if data.ndim == 2:
        n_features = data.shape[1]

        # Try to extract feature names
        if isinstance(data, pd.DataFrame):
            feature_names = data.columns.tolist()
            data = data.values
        elif feature_names is None:
            feature_names = [f"Feature {i+1}" for i in range(n_features)]

        print(f"🧪 Analyzing a dataset with {n_features} features.\n")

        for i, column in enumerate(data.T):
            name = feature_names[i]
            print(f"\n📌 Feature {i + 1}: {name}")
            analyze_distribution(column, feature_label=name, summary_report=summary_report)

        # Print final report
        print("\n📋 Summary Report:")
        for line in summary_report:
            print(f" - {line}")
        return

    # Analyze 1D feature
    feature_name = feature_label if feature_label else "Unnamed Feature"

    print(f"\n🔍 Analyzing '{feature_name}'")

    # Basic Stats
    mean = np.mean(data)
    std = np.std(data)
    cv = std / mean if mean != 0 else float('inf')
    data_skew = skew(data)
    data_kurtosis = kurtosis(data)

    # IQR and Outliers
    Q1 = np.percentile(data, 25)
    Q3 = np.percentile(data, 75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = (data < lower_bound) | (data > upper_bound)
    outlier_ratio = np.sum(outliers) / len(data)

    # Percentile Ratios
    p70 = np.percentile(data, 70)
    p99 = np.percentile(data, 99)
    p_ratio = p99 / p70 if p70 != 0 else float('inf')

    # Judgments
    judgments = {
        'CV':          ("✅ Good" if cv < 1.0 else "⚠️ High variability"),
        'Skewness':    ("✅ Symmetric" if abs(data_skew) < 0.5 else "⚠️ Skewed"),
        'Kurtosis':    ("✅ Normal-like" if -1 < data_kurtosis < 3 else "⚠️ Extreme tails"),
        'Outliers':    ("✅ Low outliers" if outlier_ratio < 0.05 else "⚠️ Too many outliers"),
        'P-Ratio':     ("✅ Stable spread" if p_ratio < 2.0 else "⚠️ Top-end dominance")
    }

    # Print Metrics & Judgments
    print(f"🔸 Mean: {mean:.4f}")
    print(f"🔸 Std Dev: {std:.4f}")
    print(f"🔸 Coefficient of Variation (CV): {cv:.4f} → {judgments['CV']}")
    print(f"🔸 Skewness: {data_skew:.4f} → {judgments['Skewness']}")
    print(f"🔸 Kurtosis: {data_kurtosis:.4f} → {judgments['Kurtosis']}")
    print(f"🔸 IQR: {IQR:.4f}")
    print(f"🔸 Outlier Ratio (Tukey's rule): {outlier_ratio * 100:.2f}% → {judgments['Outliers']}")
    print(f"🔸 99th / 70th Percentile: {p_ratio:.2f} → {judgments['P-Ratio']}")
    print()

    # Overall Readiness
    readiness = all("✅" in v for v in judgments.values())
    msg = f"✅ '{feature_name}' appears ready for training." if readiness else f"⚠️ '{feature_name}' may require preprocessing."
    print(msg + "\n")
    summary_report.append(f"{feature_name}: {'Ready ✅' if readiness else 'Needs preprocessing ⚠️'}")

    # Plots
    plt.figure(figsize=(12, 5))

    # Histogram
    plt.subplot(1, 2, 1)
    plt.hist(data, bins=40, edgecolor='black', alpha=0.7)
    plt.title(f'Histogram: {feature_name}')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.grid(True)

    # CDF
    plt.subplot(1, 2, 2)
    sorted_data = np.sort(data)
    cdf = np.arange(len(data)) / len(data)
    plt.plot(sorted_data, cdf)
    plt.title(f'CDF: {feature_name}')
    plt.xlabel('Value')
    plt.ylabel('Cumulative Probability')
    plt.grid(True)

    plt.tight_layout()
    plt.show()

def normalize(data):
    data = np.log1p(data)  # Apply log1p to all elements
    mean = np.mean(data, axis=0)  # Per feature
    std = np.std(data, axis=0)    # Per feature
    normalized = (data - mean) / std
    return normalized, mean, std

def denormalize(data, old_mean, old_std):
    restored = data * old_std + old_mean
    restored = np.expm1(restored)
    return restored

def feature_1_normalize(train_data, test_data):
    #Applying Scalars Avoiding Data Leakage

    #Applying Log Transformer to Solve Skewness of Data
    train_data = np.log1p(train_data)
    test_data = np.log1p(test_data)
    #Applying Robust Scaler to Solve Outliers Reversibly
    scaler_1 = RobustScaler()
    train_data = train_data.reshape(-1, 1)
    train_data = scaler_1.fit_transform(train_data)
    test_data = test_data.reshape(-1, 1)
    test_data = scaler_1.transform(test_data)
    scaler_2 = MinMaxScaler()
    train_data = scaler_2.fit_transform(train_data)
    test_data = scaler_2.transform(test_data)
    return train_data.flatten(), test_data.flatten() , scaler_1, scaler_2

def feature_1_denormalize(scaled_data, scaler_1, scaler_2):

    scaled_data = np.array(scaled_data).reshape(-1, 1)
    data_scaled_1 = scaler_2.inverse_transform(scaled_data)
    # Reverse RobustScaler
    data_logged = scaler_1.inverse_transform(data_scaled_1)
    # Reverse log1p
    data_original = np.expm1(data_logged)

    return data_original.flatten()


































def analyze_stats(data):
    mean = np.mean(data)
    std = np.std(data)
    cv = std / mean if mean != 0 else float('inf')
    data_skew = skew(data)
    data_kurtosis = kurtosis(data)
    Q1 = np.percentile(data, 25)
    Q3 = np.percentile(data, 75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = (data < lower_bound) | (data > upper_bound)
    outlier_ratio = np.sum(outliers) / len(data)
    p70 = np.percentile(data, 70)
    p99 = np.percentile(data, 99)
    p_ratio = p99 / p70 if p70 != 0 else float('inf')

    return {
        'Mean': mean,
        'Std Dev': std,
        'CV': cv,
        'Skewness': data_skew,
        'Kurtosis': data_kurtosis,
        'IQR': IQR,
        'Outlier Ratio': outlier_ratio,
        'Percentile Ratio': p_ratio
    }

def format_judgment(value, metric):
    if metric == 'CV':
        return "✅ Good" if value < 1.0 else "⚠️ High variability"
    if metric == 'Skewness':
        return "✅ Symmetric" if abs(value) < 0.5 else "⚠️ Skewed"
    if metric == 'Kurtosis':
        return "✅ Normal-like" if -1 < value < 3 else "⚠️ Extreme tails"
    if metric == 'Outlier Ratio':
        return "✅ Low outliers" if value < 0.05 else "⚠️ Too many outliers"
    if metric == 'Percentile Ratio':
        return "✅ Stable spread" if value < 2.0 else "⚠️ Top-end dominance"
    return ""

def compare_histograms_with_stats(data_before, data_after, feature_name='Feature', bins=40):
    stats_before = analyze_stats(data_before)
    stats_after = analyze_stats(data_after)

    plt.figure(figsize=(14, 7))

    # Before transformation histogram
    plt.subplot(2, 2, 1)
    plt.hist(data_before, bins=bins, edgecolor='black', alpha=0.7)
    plt.title(f'{feature_name} Before Transformation')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.grid(True)

    # Print stats below histogram
    plt.subplot(2, 2, 3)
    plt.axis('off')
    text_before = f"🔍 {feature_name} Before Transformation Stats\n"
    for key, value in stats_before.items():
        judgment = format_judgment(value, key)
        if key == 'Outlier Ratio':
            value_str = f"{value*100:.2f}%"
        elif key == 'Skewness' or key == 'Kurtosis' or key == 'CV' or key == 'Percentile Ratio':
            value_str = f"{value:.4f}"
        else:
            value_str = f"{value:.4f}"
        text_before += f"🔸 {key}: {value_str} → {judgment}\n"
    plt.text(0, 0.5, text_before, fontsize=10, va='center')

    # After transformation histogram
    plt.subplot(2, 2, 2)
    plt.hist(data_after, bins=bins, edgecolor='black', alpha=0.7, color='orange')
    plt.title(f'{feature_name} After Transformation')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.grid(True)

    # Print stats below histogram
    plt.subplot(2, 2, 4)
    plt.axis('off')
    text_after = f"🔍 {feature_name} After Transformation Stats\n"
    for key, value in stats_after.items():
        judgment = format_judgment(value, key)
        if key == 'Outlier Ratio':
            value_str = f"{value*100:.2f}%"
        elif key == 'Skewness' or key == 'Kurtosis' or key == 'CV' or key == 'Percentile Ratio':
            value_str = f"{value:.4f}"
        else:
            value_str = f"{value:.4f}"
        text_after += f"🔸 {key}: {value_str} → {judgment}\n"
    plt.text(0, 0.5, text_after, fontsize=10, va='center')

    plt.tight_layout()
    plt.show()


import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import zscore


def plot_single_feature_vs_target_with_outliers(X, y, threshold=3):
    """
    Plot the single feature vs target,
    mark outliers based on Z-score on the feature.
    """
    feature_data = X.flatten()  # ensure 1D array

    # Calculate IQR
    Q1 = np.percentile(feature_data, 25)
    Q3 = np.percentile(feature_data, 75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Detect outliers using IQR bounds
    outlier_mask = (feature_data < lower_bound) | (feature_data > upper_bound)
    outlier_data = feature_data[outlier_mask]

    # Debug / troubleshooting info
    print(f"Feature data stats:")
    print(
        f"  Min: {feature_data.min():.4f}, Q1: {Q1:.4f}, Median: {np.median(feature_data):.4f}, Q3: {Q3:.4f}, Max: {feature_data.max():.4f}")
    print(f"  IQR: {IQR:.4f}")
    print(f"  Lower bound: {lower_bound:.4f}")
    print(f"  Upper bound: {upper_bound:.4f}")
    print(f"Outliers detected: {np.sum(outlier_mask)} points ({np.sum(outlier_mask) / len(feature_data) * 100:.2f}%)")

    # Plot
    plt.figure(figsize=(9, 6))
    plt.scatter(feature_data[~outlier_mask], y[~outlier_mask], label='Normal points', alpha=0.6, s=40, color='blue')
    plt.scatter(feature_data[outlier_mask], y[outlier_mask], label='Outliers', s=100,
                facecolors='none', edgecolors='red', linewidth=2, alpha=0.7)

    plt.xlabel('Feature')
    plt.ylabel('Target y')
    plt.title('Feature vs Target with IQR-based Outliers Marked')
    plt.legend()
    plt.grid(True)
    plt.show()

    return np.where(outlier_mask)[0]


def plot_timeseries_with_feature_outliers(feature_data, y_data):
    """
    Plot time series of a feature and y on the same plot,
    highlight feature outliers detected by IQR in red,
    normal points in blue.

    Parameters:
    - feature_data: 1D array-like of feature values (time series)
    - y_data: 1D array-like of y values (time series), same length as feature_data
    """
    feature_data = np.array(feature_data).flatten()
    y_data = np.array(y_data).flatten()
    n = len(feature_data)
    time = np.arange(n)

    # IQR for outlier detection on feature_data
    Q1 = np.percentile(feature_data, 25)
    Q3 = np.percentile(feature_data, 75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outlier_mask = (feature_data < lower_bound) | (feature_data > upper_bound)

    print(f"Detected {np.sum(outlier_mask)} outliers out of {n} points")

    plt.figure(figsize=(12, 6))

    # Plot feature data points: normal and outliers with different colors and markers
    plt.plot(time[~outlier_mask], feature_data[~outlier_mask], 'bo-', label='Feature (normal)', markersize=4)
    plt.plot(time[outlier_mask], feature_data[outlier_mask], 'ro', label='Feature (outliers)', markersize=8,
             markerfacecolor='none', markeredgewidth=2)

    # Plot y data as a line
    plt.plot(time, y_data, 'g-', label='Y data', alpha=0.7)

    # Mark IQR bounds as horizontal lines for reference
    plt.axhline(lower_bound, color='red', linestyle='--', alpha=0.5, label='IQR lower bound')
    plt.axhline(upper_bound, color='red', linestyle='--', alpha=0.5, label='IQR upper bound')

    plt.xlabel('Time (index)')
    plt.ylabel('Value')
    plt.title('Time Series Plot with Feature Outliers Highlighted (IQR Method)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return np.where(outlier_mask)[0]


def Preprocess(X_train, X_test, y_train, y_test, visualize = False):
    """
    Normalize each feature in X_train and X_test independently using feature_1_normalize,
    normalize y_train and y_test, visualize histograms, and return normalized data.

    Returns:
        X_train_normalized, X_test_normalized, y_train_normalized, y_test_normalized
    """

    # Extract individual features
    OPCP_train, OPCP_test = X_train[:, 0], X_test[:, 0]
    HPCP_train, HPCP_test = X_train[:, 1], X_test[:, 1]
    LPCP_train, LPCP_test = X_train[:, 2], X_test[:, 2]
    CPCP_train, CPCP_test = X_train[:, 3], X_test[:, 3]
    ACPCP_train, ACPCP_test = X_train[:, 4], X_test[:, 4]

    # Normalize each feature independently
    new_OPCP_train, new_OPCP_test, _, _ = feature_1_normalize(OPCP_train, OPCP_test)
    new_HPCP_train, new_HPCP_test, _, _ = feature_1_normalize(HPCP_train, HPCP_test)
    new_LPCP_train, new_LPCP_test, _, _ = feature_1_normalize(LPCP_train, LPCP_test)
    new_CPCP_train, new_CPCP_test, _, _ = feature_1_normalize(CPCP_train, CPCP_test)
    new_ACPCP_train, new_ACPCP_test, _, _ = feature_1_normalize(ACPCP_train, ACPCP_test)

    # Stack features back horizontally
    train_features = [
        new_OPCP_train.reshape(-1, 1),
        new_HPCP_train.reshape(-1, 1),
        new_LPCP_train.reshape(-1, 1),
        new_CPCP_train.reshape(-1, 1),
        new_ACPCP_train.reshape(-1, 1),
    ]

    test_features = [
        new_OPCP_test.reshape(-1, 1),
        new_HPCP_test.reshape(-1, 1),
        new_LPCP_test.reshape(-1, 1),
        new_CPCP_test.reshape(-1, 1),
        new_ACPCP_test.reshape(-1, 1),
    ]

    X_train_normalized = np.hstack(train_features)
    X_test_normalized = np.hstack(test_features)

    # Normalize target
    y_train_normalized, y_test_normalized, scaler_y_1, scaler_y_2 = feature_1_normalize(y_train, y_test)

    # Plot before and after normalization histograms (12 subplots)
    feature_names = ["OPCP", "HPCP", "LPCP", "CPCP", "ACPCP", "Target (y)"]
    raw_train_features = [OPCP_train, HPCP_train, LPCP_train, CPCP_train, ACPCP_train, y_train]
    norm_train_features = [new_OPCP_train, new_HPCP_train, new_LPCP_train, new_CPCP_train, new_ACPCP_train,
                           y_train_normalized]

    fig, axs = plt.subplots(6, 2, figsize=(12, 18))
    fig.suptitle("Before and After Normalization Histograms (Train Data)", fontsize=16)

    for i, feature_name in enumerate(feature_names):
        # Before normalization
        axs[i, 0].hist(raw_train_features[i], bins=30, color='blue', alpha=0.7)
        axs[i, 0].set_title(f"{feature_name} - Before Norm")
        axs[i, 0].grid(True)

        # After normalization
        axs[i, 1].hist(norm_train_features[i], bins=30, color='green', alpha=0.7)
        axs[i, 1].set_title(f"{feature_name} - After Norm")
        axs[i, 1].grid(True)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    if visualize:
        plt.show()

    return X_train_normalized, X_test_normalized, y_train_normalized, y_test_normalized , scaler_y_1, scaler_y_2

