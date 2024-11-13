import pandas as pd
import numpy as np


def load_csv(file_path):
    # Load the CSV file
    df = pd.read_csv(file_path)

    # Drop "Time" and "Frame#" columns if they exist
    df = df.drop(columns=["Time", "Frame#"], errors="ignore")

    return df


def compute_one_sided_fft(df, sampling_rate=120):
    """
    Computes the one-sided FFT (Fast Fourier Transform) for each column in the DataFrame, focusing on
    a frequency range from 2 Hz to 20 Hz. This approach isolates meaningful frequency components by examining
    only the positive half of the frequency spectrum (one-sided FFT), which contains all necessary information
    for real-valued signals.

    Parameters:
    - df (DataFrame): Input DataFrame where each column contains time-series data.
    - sampling_rate (float): Sampling rate of the data in Hz. Default is 120 Hz.

    Returns:
    - fft_df (DataFrame): DataFrame with the same column names, each containing the one-sided FFT values
      interpolated to the specified frequency range (2-20 Hz).
    """

    fs = sampling_rate
    freqs = np.fft.rfftfreq(df.shape[0], d=1/fs)
    freq_mask = (freqs >= 2) & (freqs <= 20)
    fft_df = pd.DataFrame(columns=df.columns)

    for column in df.columns:
        fft_values = np.fft.rfft(df[column])

        fft_magnitude_interp = np.interp(
            np.linspace(2, 20, num=50),
            freqs[freq_mask],
            np.abs(fft_values)[freq_mask]
        )
        fft_df[column] = fft_magnitude_interp

    return fft_df


def calc_features(f):
    fftav = []

    for idx, row in f.iterrows():
        fftav.append(np.mean(row))

    return fftav


def process_all_files(label_df):
    """
    Iterates over filenames in lbl.file, computes features for each file, 
    and concatenates the results into a NumPy array.

    Parameters:
    - label_df (DataFrame): DataFrame containing a 'file' column with filenames.

    Returns:
    - features_array (np.ndarray): 2D array with one row per file and columns representing the features.
    """
    features_list = []

    for file_path in label_df.file:
        df = load_csv(file_path)
        fft_df = compute_one_sided_fft(df)
        fft_features = calc_features(fft_df)
        features_list.append(fft_features)

    features_array = np.array(features_list)

    return features_array


def create_labeled_dataset(lbl, features_array):
    """
    Combines features with selected columns from labels to create a structured, labeled dataset.

    Parameters:
    - lbl (DataFrame): DataFrame containing labels and other metadata for each sample.
    - features_array (np.ndarray): Array with feature vectors (one row per sample).

    Returns:
    - labeled_df (DataFrame): A DataFrame combining features and selected label columns.
    """
    # Convert features array to DataFrame
    features_df = pd.DataFrame(features_array, columns=[
                               f"Feature_{i+1}" for i in range(features_array.shape[1])])

    # Select only the desired columns ('file', 'label_yn', 'label_raw') from lbl
    lbl_selected = lbl[['file', 'label_yn', 'label_raw']]

    # Concatenate the selected columns with features
    labeled_df = pd.concat(
        [lbl_selected.reset_index(drop=True), features_df], axis=1)

    return labeled_df
