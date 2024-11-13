import numpy as np
from pylsl import StreamInlet, resolve_byprop
import pandas as pd
import time
from scipy.fft import fft

# Set parameters
sampling_interval = 1  # 0.1 second intervals
sampling_rate = 256  # Muse 2016 EEG sampling rate (may vary based on device)

# Frequency bands (Hz)
freq_bands = {
    "delta": (0.5, 4),
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta": (13, 30),
    "gamma": (30, 50)
}

# Resolve the Muse stream
print("Looking for an EEG stream...")
streams = resolve_byprop('type', 'EEG', timeout=10)
if not streams:
    raise RuntimeError("Can't find EEG stream.")
inlet = StreamInlet(streams[0])

# Initialize data collection
eeg_data = []

# Define column names
column_order = ['timestamp'] + [f"{ch}_{band}" for ch in ['TP9', 'AF7', 'AF8', 'TP10', 'AUX'] for band in freq_bands.keys()]

# Start data collection
print("Starting data collection...")
start_time = time.time()
header_written = False  # To track if header has been written

while True:
    try:
        # Fetch the sample and timestamp
        sample, timestamp = inlet.pull_sample()

        # Append timestamp and sample
        eeg_data.append([timestamp] + sample)

        # Save every 0.1 seconds
        if time.time() - start_time >= sampling_interval:
            # Convert to DataFrame
            df = pd.DataFrame(eeg_data, columns=['timestamp', 'TP9', 'AF7', 'AF8', 'TP10', 'AUX'])

            # Perform FFT for each channel and store results
            fft_data = {'timestamp': timestamp}  # Use the last timestamp of this interval
            for ch in ['TP9', 'AF7', 'AF8', 'TP10', 'AUX']:
                # Compute FFT
                yf = fft(df[ch].values)
                xf = np.fft.fftfreq(len(yf), 1 / sampling_rate)

                # Filter FFT data into frequency bands and save to dictionary
                for band, (low, high) in freq_bands.items():
                    band_power = np.sum(np.abs(yf[(xf >= low) & (xf <= high)]))
                    fft_data[f"{ch}_{band}"] = band_power
            
            # Convert to DataFrame with proper column order
            result_df = pd.DataFrame([fft_data], columns=column_order)

            # Save to CSV with header written only once
            result_df.to_csv("muse_fft_data.csv", mode='a', header=not header_written, index=False)
            header_written = True  # Ensure header is only written the first time

            # Reset for next interval
            eeg_data.clear()
            start_time = time.time()
    
    except KeyboardInterrupt:
        print("Data collection stopped.")
        break

