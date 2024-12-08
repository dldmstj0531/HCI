## final_src.py
import pandas as pd
import numpy as np
from scipy.fft import fft
import mne
from mne.preprocessing import ICA
from scipy.signal import find_peaks
from collections import Counter
import matplotlib.pyplot as plt
import chardet
import os
import re 


# 자연 정렬 함수(숫자 번호 뒤섞이는거 방지)
def natural_sort_key(filename):
    return [int(text) if text.isdigit() else text.lower() for text in re.split('(\d+)', filename)]


## EEG DATA 전처리(None값 처리, ICA, FFT변환 등)

# eeg_data를 qnum기준으로 슬라이싱
def process_and_slice_eeg_data(input_folder, output_folder):
    # 저장 디렉토리가 없으면 생성
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 폴더 내 모든 파일 처리
    for input_file in os.listdir(input_folder):
        if "eeg_data" not in input_file:
            print(f"Skipping file (not an EEG data file): {input_file}")
            continue

        input_file_path = os.path.join(input_folder, input_file)
        try:
            df = pd.read_csv(input_file_path, on_bad_lines='skip')  
        except Exception as e:
            print(f"Error reading file {input_file}: {e}")
            continue

        if 'RightAux' in df.columns:
            df = df.drop('RightAux', axis=1)
        df = df[(df != -1000).all(axis=1)]

        header_row = df.iloc[:0]

        # 'qnum' 값을 기준으로 슬라이싱
        if 'qnum' not in df.columns:
            print(f"Skipping file (no 'qnum' column found): {input_file}")
            continue

        unique_qnums = df['qnum'].unique()

        for qnum in unique_qnums:
            if qnum == "qnum":
                print(f"Skipping slice with 'qnum' = 'qnum' for file: {input_file}")
                continue

            sliced_df = df[df['qnum'] == qnum]

            # 원본 첫 번째 행을 추가하여 데이터 구성
            combined_df = pd.concat([header_row, sliced_df], ignore_index=True)
            # 새로운 파일명 생성
            base_filename = os.path.splitext(input_file)[0]  
            output_file = os.path.join(output_folder, f"{base_filename}_{qnum}.csv")
            # 데이터 저장
            combined_df.to_csv(output_file, index=False)
            print(f"Saved: {output_file}")


# 주파수 대역별 파워 계산 함수
def calculate_band_powers(signal, sampling_rate, freq_bands):
    yf = fft(signal)
    xf = np.fft.fftfreq(len(yf), 1 / sampling_rate)
    band_powers = {}
    for band, (low, high) in freq_bands.items():
        band_power = np.sum(np.abs(yf[(xf >= low) & (xf <= high)]))
        band_powers[band] = band_power
    return band_powers


# EEG data => ICA, FFT
def preprocess_eeg(input_folder, output_folder, sampling_rate=100, use_ica=False):
    freq_bands = {
        "delta": (0.5, 4),
        "theta": (4, 8),
        "alpha": (8, 13),
        "beta": (13, 30),
        "gamma": (30, 50)
    }
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for input_file in os.listdir(input_folder):
        input_file_path = os.path.join(input_folder, input_file)
        try:
            columns = ['timestamp', 'qnum', 'TP9', 'AF7', 'AF8', 'TP10']
            data = pd.read_csv(input_file_path, names=columns, header=0)
        except Exception as e:
            #print(f"Error reading file {input_file}: {e}")
            continue

        eeg_channels = ['TP9', 'AF7', 'AF8', 'TP10']
        timestamps = data['timestamp']
        eeg_data = data[eeg_channels].apply(pd.to_numeric, errors='coerce').dropna()

        # MNE 데이터 변환
        info = mne.create_info(ch_names=eeg_channels, sfreq=sampling_rate, ch_types='eeg')
        raw = mne.io.RawArray(eeg_data.T.values, info)

        # ICA 적용 여부
        if use_ica:
            ica = ICA(n_components=len(eeg_channels), random_state=97, max_iter=800)
            ica.fit(raw)
            raw = ica.apply(raw)  # 잡음 제거된 데이터

        clean_data = raw.get_data().T
        eeg_data = pd.DataFrame(clean_data, columns=eeg_channels)

        fft_results = []
        window_size = int(sampling_rate * 0.5)
        step_size = window_size

        for start_idx in range(0, len(eeg_data) - window_size, step_size):
            end_idx = start_idx + window_size
            row_result = {'timestamp': timestamps.iloc[start_idx]}
            for channel in eeg_channels:
                signal = eeg_data[channel].iloc[start_idx:end_idx].values
                band_powers = calculate_band_powers(signal, sampling_rate, freq_bands)
                for band, power in band_powers.items():
                    row_result[f"{channel}_{band}"] = power
            fft_results.append(row_result)

        fft_df = pd.DataFrame(fft_results)
        base_filename = os.path.splitext(input_file)[0]
        output_file = os.path.join(output_folder, f"{base_filename}_fft.csv")
        fft_df.to_csv(output_file, index=False)
        print(f"Saved FFT result: {output_file}")

# Ratio 계산
def calculate_ratios(df, electrodes):
    for electrode in electrodes:
        alpha_col = f"{electrode}_alpha"
        beta_col = f"{electrode}_beta"
        theta_col = f"{electrode}_theta"
        gamma_col = f"{electrode}_gamma"

        # Beta/Alpha 비율
        df[f"{electrode}_Beta_Alpha_Ratio"] = df[beta_col]/df[alpha_col]
        # Theta/Beta 비율
        df[f"{electrode}_Theta_Beta_Ratio"] = df[theta_col] / df[beta_col]
        # Gamma/Beta 비율
        df[f"{electrode}_Gamma_Beta_Ratio"] = df[gamma_col] / df[beta_col]

    return df



# 확신 상태 점수 계산 함수 정의
def calculate_confidence(row):
     # Beta/Alpha와 Alpha/Theta 평균
    beta_alpha_avg = (row['AF7_Beta_Alpha_Ratio'] + row['AF8_Beta_Alpha_Ratio']+row['TP9_Beta_Alpha_Ratio'] + row['TP10_Beta_Alpha_Ratio']) / 4
    theta_beta_avg = (row['AF7_Theta_Beta_Ratio'] + row['AF8_Theta_Beta_Ratio']+row['TP9_Theta_Beta_Ratio'] + row['TP10_Theta_Beta_Ratio']) / 4
    gamma_beta_avg = (row['AF7_Gamma_Beta_Ratio'] + row['AF8_Gamma_Beta_Ratio']+row['TP9_Gamma_Beta_Ratio'] + row['TP10_Gamma_Beta_Ratio']) / 4

    # 종합 확신 점수 
    confidence_score = 0.49 * beta_alpha_avg + 0.67 * theta_beta_avg +  0.33 * gamma_beta_avg
    return confidence_score
## 데이터 분석
'''
# Ratio 계산
def calculate_ratios(df, electrodes):
    for electrode in electrodes:
        alpha_col = f"{electrode}_alpha"
        beta_col = f"{electrode}_beta"
        theta_col = f"{electrode}_theta"

        # Beta/Alpha 비율
        df[f"{electrode}_Beta_Alpha_Ratio"] = df[beta_col]/df[alpha_col]
        # Theta/Beta 비율
        df[f"{electrode}_Theta_Beta_Ratio"] = df[theta_col] / df[beta_col]

    return df


# 확신 상태 점수 계산 함수 정의
def calculate_confidence(row):
    # Beta/Alpha와 Alpha/Theta 평균
    beta_alpha_avg = (row['AF7_Beta_Alpha_Ratio'] + row['AF8_Beta_Alpha_Ratio']) / 2
    theta_Beta_avg = (row['AF7_Theta_Beta_Ratio'] + row['AF8_Theta_Beta_Ratio']) / 2

    # 종합 확신 점수 (가중 평균)
    confidence_score = 0.5 * beta_alpha_avg + 0.5 * theta_Beta_avg
    return confidence_score
'''

# Confidence Score 기반 피크 및 선정 구간 탐지 (Max 값 기준)
def detect_and_define_intervals(data, column, interval_width=1, height_threshold=0.1):
    # peak 찾기 
    peaks, properties = find_peaks(data[column], height=height_threshold)
    if len(peaks) == 0:
        max_value = data[column].max()
        max_peak_index = data[column].idxmax()
    else:
        peak_values = data[column].iloc[peaks]
        max_peak_index = peak_values.idxmax() 
        max_value = data[column].iloc[max_peak_index]
    #집중 구간 계산
    time = data['timestamp'].iloc[max_peak_index]
    start_time = time - interval_width / 2
    end_time = time + interval_width / 2

    return start_time, end_time, max_value


# Confidence Score 및 피크, 구간 시각화
def plot_confidence_and_single_interval(data, column, interval, output_file=None):
    plt.figure(figsize=(12, 6))
    plt.plot(data['timestamp'], data[column], label='Confidence Score', color='blue')

    start_time, end_time = interval
    plt.axvspan(start_time, end_time, color='yellow', alpha=0.3, label='Selected Interval')

    plt.xlabel('Timestamp')
    plt.ylabel(column)
    plt.title(f'{column} and Selected Interval')

    # 중복된 레이블 제거
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), loc='best')

    # 이미지 저장
    if output_file:
        plt.savefig(output_file.replace('.png', '.jpg'), format='jpg', dpi=100)
        print(f"Plot saved as {output_file}")
    
    plt.close('all')


## Gaze_data와 연결

# Gaze 데이터에서 특정 구간의 데이터를 필터링
def filter_gaze_data_by_intervals(gaze_data, intervals):
    selected_rows = []
    for start_time, end_time in intervals:
        rows_in_interval = gaze_data[(gaze_data['timestamp'] >= start_time) &
                                     (gaze_data['timestamp'] <= end_time)]
        selected_rows.append(rows_in_interval)

    if selected_rows:
        final_interval = pd.concat(selected_rows, ignore_index=True)
        return final_interval['state'].values
    return []


# 가장 많이 등장한 Gaze 값을 계산
from collections import Counter

def get_most_common_gaze_value(gaze_data_value):
    counter = Counter(gaze_data_value)
    most_common = counter.most_common()

    if not most_common:
        return None, 0  

    # 'blinking' 값을 제거한 목록 생성
    filtered_common = [item for item in most_common if item[0] != "blinking"]
    if not filtered_common:
        return None, 0  

    max_count = filtered_common[0][1]
    max_items = [item for item, count in filtered_common if count == max_count]

    if len(max_items) > 1:
        # 가장 많이 등장한 값이 여러 개인 경우 평균 계산
        print(f"경고: 가장 많이 등장한 값이 여러 개입니다: {max_items} (각 {max_count}회 등장)")
        average = round(sum(float(item[0]) for item in max_items) / len(max_items), 1)
        return average, max_count
    else:
        return float(filtered_common[0][0]), max_count



# gaze data 상태를 판별
def determine_g_state(df):
    if not {'horizontal_ratio', 'vertical_ratio', 'blinking_ratio'}.issubset(df.columns):
        raise ValueError("데이터 프레임에 'horizontal_ratio', 'vertical_ratio', 'blinking_ratio' 열이 있어야 합니다.")

    def get_f_state(horizontal_ratio, vertical_ratio, blinking_ratio):
        if blinking_ratio > 4.0:  # 깜빡임 상태 판별
            return "blinking"
        elif horizontal_ratio <= 0.53:  
            return "5"
        elif horizontal_ratio <= 0.598:  
            return "4"
        elif horizontal_ratio >= 0.635: 
            return "2"
        elif horizontal_ratio >= 0.691:  
            return "1"
        else:  # 중앙 상태 판별
            return "3"
    # state 열 생성
    df['state'] = df.apply(lambda x: get_f_state(x['horizontal_ratio'], x['vertical_ratio'], x['blinking_ratio']),axis=1)
    return df


# Main
if __name__ == "__main__":
    # Step 1: Process and slice EEG data
    process_and_slice_eeg_data("data_src", "data_output")
    
    # Step 2: Preprocess EEG and perform FFT
    preprocess_eeg("data_output", "data_output",sampling_rate=100, use_ica=True)
    
    # Step 3: Analyze EEG data for confidence intervals
    input_folder = "data_output"
    output_folder = "data_output"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    electrodes = ['TP9', 'AF7', 'AF8', 'TP10']
    grouped_intervals = {}
    con_score = []
    for input_file in sorted(os.listdir(input_folder), key=natural_sort_key):  # 자연 정렬 적용
        if not input_file.endswith("_fft.csv"):
            continue
        input_path = os.path.join(input_folder, input_file)
        data = pd.read_csv(input_path)
        data = calculate_ratios(data, electrodes)
        data['Confidence_Score'] = data.apply(calculate_confidence, axis=1)
        start_time, end_time, max_value = detect_and_define_intervals(data, 'Confidence_Score', interval_width=1, height_threshold=0.1)
        con_score.append(max_value)
        output_plot_file = os.path.join(output_folder, f"{os.path.splitext(input_file)[0]}_confidence_plot.png")
        plot_confidence_and_single_interval(
            data,
            column='Confidence_Score',
            interval=(start_time, end_time),
            output_file=output_plot_file
        )
        base_key = "_".join(input_file.split("_")[:2])
        if base_key not in grouped_intervals:
            grouped_intervals[base_key] = []
        grouped_intervals[base_key].append({'start_time': start_time, 'end_time': end_time})
        base_filename = os.path.splitext(input_file)[0]
        output_path_data = os.path.join(output_folder, f"{base_filename}_processed.csv")
        data.to_csv(output_path_data, index=False)
    for base_key, intervals in grouped_intervals.items():
        intervals_df = pd.DataFrame(intervals)
        intervals_df.to_csv(os.path.join(output_folder, f"{base_key}_intervals.csv"), index=False)

    # Step 4: Process gaze data
    gaze_data = pd.read_csv("data_src/gaze_data.csv").dropna()
    gaze_data = determine_g_state(gaze_data)
    gaze_data.to_csv("data_src/gaze_data_processed.csv")

    # Step 5: Link gaze data with intervals
    eeg_intervals_file = "data_output/eeg_data_intervals.csv"
    gaze_analysis_file = "data_src/gaze_data_processed.csv"
    eeg_intervals = pd.read_csv(eeg_intervals_file)
    gaze_data = pd.read_csv(gaze_analysis_file)
    results = []
    for _, interval in eeg_intervals.iterrows():
        start_time, end_time = interval['start_time'], interval['end_time']
        filtered_gaze_data = filter_gaze_data_by_intervals(gaze_data, [(start_time, end_time)])
        most_common_state, count = get_most_common_gaze_value(filtered_gaze_data)
        results.append({'start_time': start_time, 'end_time': end_time, 'most_common_state': most_common_state, 'count': count})
    results_df = pd.DataFrame(results)
    results_df.to_csv("gaze_analysis_results.csv", index=False)

    # Step 6: Update response times with gaze data
    response_data = pd.read_csv("data_src/response_times.csv").rename(columns={'qnum': 'qnum', 'response': 'ans_num', 'res_time': 'response_time'})
    gaze_analysis_data = pd.read_csv("gaze_analysis_results.csv")
    response_data['eeg_num'] = gaze_analysis_data['most_common_state'].fillna(0).astype(int)
    response_data['con_score'] = con_score
    response_data.to_csv("final_report.csv", index=False, encoding='utf-8-sig')
    print("Updated response times saved to final_report.csv")