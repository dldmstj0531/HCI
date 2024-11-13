import numpy as np
from pylsl import StreamInlet, resolve_byprop
import pandas as pd
import time
from scipy.fft import fft

# Set parameters
sampling_interval = 1  
sampling_rate = 256  

freq_bands = {
    "delta": (0.5, 4),
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta": (13, 30),
    "gamma": (30, 50)
}

print("Looking for an EEG stream...")
streams = resolve_byprop('type', 'EEG', timeout=10)
if not streams:
    raise RuntimeError("Can't find EEG stream.")
inlet = StreamInlet(streams[0])

eeg_data = []

column_order = ['timestamp'] + [f"{ch}_{band}" for ch in ['TP9', 'AF7', 'AF8', 'TP10', 'AUX'] for band in freq_bands.keys()]

print("Starting data collection...")
start_time = time.time()
header_written = False  

while True:
    try:
        sample, timestamp = inlet.pull_sample()

        eeg_data.append([timestamp] + sample)

        if time.time() - start_time >= sampling_interval:
            df = pd.DataFrame(eeg_data, columns=['timestamp', 'TP9', 'AF7', 'AF8', 'TP10', 'AUX'])

            fft_data = {'timestamp': timestamp}  
            for ch in ['TP9', 'AF7', 'AF8', 'TP10', 'AUX']:
                yf = fft(df[ch].values)
                xf = np.fft.fftfreq(len(yf), 1 / sampling_rate)

                for band, (low, high) in freq_bands.items():
                    band_power = np.sum(np.abs(yf[(xf >= low) & (xf <= high)]))
                    fft_data[f"{ch}_{band}"] = band_power
            
            result_df = pd.DataFrame([fft_data], columns=column_order)

            result_df.to_csv("muse_fft_data.csv", mode='a', header=not header_written, index=False)
            header_written = True  

           
            eeg_data.clear()
            start_time = time.time()
    
    except KeyboardInterrupt:
        print("Data collection stopped.")
        break

