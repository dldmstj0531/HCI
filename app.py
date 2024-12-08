from flask import Flask, render_template, request, redirect, url_for, send_file, jsonify, session
import multiprocessing
import time
import csv
import cv2
import numpy as np
from pylsl import StreamInlet, resolve_byprop
from gaze_tracking import GazeTracking
import os
import subprocess
import pandas as pd
from fpdf import FPDF
import threading
import shutil

app = Flask(__name__)
app.secret_key = 'your_secret_key'

users = {
    "admin": "1234",
    "user1": "5678"
}

questions = [
    "시간이 걸리는 일, 놀이는 싫증을 빨리 내고 새로운 놀이, 활동을 원한다",  # 6초 e   
    "다른 아이들이 생각하지도 않은 엉뚱한 행동이나 상상을 하기 어렵다",  # 5초 s
    "친한 사람이나 친구가 없는 모임에 가도 잘 적응하는 편이다", # 5초 e
    "남의 지시에 따르기보다는 자신의 마음에 따라 행동하는 것이 좋다", # 6초 p
    "주변 사람들의 얼굴, 키나 다른 특징들을 잘 기억한다", # 5초 s
    "음식, 옷을 선택할 때 쉽게 결정을 내리지 못한다", # 4초 p
    "주변 일에 호기심이 많고 새로운 일이 생겨도 잘 적응한다", # 5초 f
    "다른 친구들이나 어른들이 내 행동을 어떻게 생각할지 신경이 쓰인다", # 6초 f
]
choices = ["매우 아니다", "아니다", "보통이다", "그렇다", "매우 그렇다"]

choice_to_value = {
    "매우 아니다": 1,
    "아니다": 2,
    "보통이다": 3,
    "그렇다": 4,
    "매우 그렇다": 5
}

gaze_process = None
muse_process = None

def muse_data_collection(data_collection_active, ready_flag, question_number):
    print("Starting Muse EEG data collection...")

    data_src_dir = os.path.join(os.getcwd(), 'data_src')
    if not os.path.exists(data_src_dir): 
        os.makedirs(data_src_dir)

    try:
        inlet = None
        retry_count = 0
        while retry_count < 5 and inlet is None:
            print(f"Resolving EEG stream... (Attempt {retry_count + 1})")
            streams = resolve_byprop('type', 'EEG', timeout=10)
            if len(streams) > 0:
                inlet = StreamInlet(streams[0], max_buflen=360) # 버퍼 크기 조정 완료 
                print("EEG stream resolved successfully.")
            else:
                print("No EEG stream found. Retrying...")
                retry_count += 1
                time.sleep(5)

        if inlet is None:
            print("Failed to resolve EEG stream after several attempts.")
            return
        
         # muse 완료 알림
        ready_flag.set()


        eeg_file_path = os.path.join(data_src_dir, 'eeg_data.csv')
        with open(eeg_file_path, 'a', newline='') as eeg_file:
            eeg_writer = csv.writer(eeg_file)
            eeg_writer.writerow(["timestamp", "qnum", "TP9", "AF7", "AF8", "TP10","RightAux"])

            while not data_collection_active.value:
                time.sleep(0.01)  # 데이터 수집 완료 대기 

            while data_collection_active.value:
                sample, _ = inlet.pull_sample(timeout=1.0)

                # timestamp 현재 시간 반영 수정
                current_time = float(time.time())

                if sample is not None:
                    eeg_writer.writerow([current_time, question_number.value] + sample)
                time.sleep(0.01) # 0.01초마다 수집 

    except Exception as e:
        print(f"Error occurred during Muse EEG data collection: {e}")

def gaze_data_collection(data_collection_active, ready_flag, gaze_data, question_number):
    print("Starting Gaze Tracking data collection...")
    gaze = GazeTracking()
    webcam = cv2.VideoCapture(0)

    if not webcam.isOpened():
        print("Could not open webcam.")
        return

    # gaze 완료 알림
    ready_flag.set()

    try:
        while not data_collection_active.value:
            time.sleep(0.01)  # 데이터 수집 완료 대기

        while data_collection_active.value:
            _, frame = webcam.read()
            if frame is not None:
                gaze.refresh(frame)
                '''
                text = ""
                if gaze.is_blinking():
                    text = "깜박임"
                elif gaze.is_right():
                    text = "오른쪽을 바라보고 있습니다"
                elif gaze.is_left():
                    text = "왼쪽을 바라보고 있습니다"
                elif gaze.is_center():
                    text = "중앙을 바라보고 있습니다"
                '''
                left_pupil_coords = gaze.pupil_left_coords() 
                right_pupil_coords = gaze.pupil_right_coords()
                
                horizontal_ratio = gaze.horizontal_ratio()
                vertical_ratio = gaze.vertical_ratio()
                blinking_ratio = gaze.blinking_ratio()
                current_time = float(time.time())  # timestamp 현재 시간 수정
                gaze_data.append([current_time, question_number.value, horizontal_ratio, vertical_ratio, blinking_ratio])
            time.sleep(0.45)  # gaze 0.2초마다 수집
    finally:
        webcam.release()

def save_data_to_csv(gaze_data, response_times):
    data_src_dir = os.path.join(os.getcwd(), 'data_src')
    if not os.path.exists(data_src_dir):
        os.makedirs(data_src_dir)

    gaze_file_path = os.path.join(data_src_dir, 'gaze_data.csv')
    with open(gaze_file_path, 'w', newline='') as gaze_file:
        gaze_writer = csv.writer(gaze_file)
        gaze_writer.writerow(["timestamp", "qnum", "horizontal_ratio", "vertical_ratio", "blinking_ratio"])
        gaze_writer.writerows(gaze_data)

    
    response_file_path = os.path.join(data_src_dir, 'response_times.csv')
    with open(response_file_path, 'w', newline='') as response_file:
        response_writer = csv.writer(response_file)
        response_writer.writerow(["qnum", "response", "res_time"])
        for response in response_times:
            question_number, response_text, response_time = response
            response_value = choice_to_value.get(response_text, 0)
            response_writer.writerow([question_number, response_value, response_time])

def start_data_collection(data_collection_active, eeg_data, gaze_data, question_number):
    data_collection_active.value = False  # 데이터 수집은 준비 후 시작

    global gaze_process, muse_process

    muse_ready = multiprocessing.Event()
    gaze_ready = multiprocessing.Event()

    gaze_process = multiprocessing.Process(target=gaze_data_collection, args=(data_collection_active, gaze_ready, gaze_data, question_number))
    muse_process = multiprocessing.Process(target=muse_data_collection, args=(data_collection_active, muse_ready, question_number))

    gaze_process.start()
    muse_process.start()

    # 동시 수집 상태 완료 대기
    muse_ready.wait()
    gaze_ready.wait()

    # 시작 시간 비슷하게 수정 
    common_start_time = time.time()
    data_collection_active.value = True

    print(f"Both Muse and Gaze tracking are ready. Starting survey at timestamp {common_start_time}")

def stop_data_collection():
    global gaze_process, muse_process
    data_collection_active.value = False

    if gaze_process is not None:
        gaze_process.join()
    
    if muse_process is not None:
        muse_process.join()
        
    data_output_path = os.path.join(os.getcwd(), "data_src", "data_output")
    if os.path.exists(data_output_path) and not os.listdir(data_output_path):
        shutil.rmtree(data_output_path)
        print(f"Deleted empth directory: {data_output_path}")
    
    # 데이터 수집 완료 시 final_src.py 실행
    save_data_to_csv(gaze_data, response_times)
    try:
        # final_src.py 실행
        final_src_path = os.path.join(os.getcwd(), "final_src.py")
        subprocess.run(
            ["python", final_src_path],
            cwd=os.path.join(os.getcwd(), 'data_src'),
            check=True
        )
    except subprocess.CalledProcessError as e:
        print(f"Error occurred while running final_src.py: {e}")

def analyze_and_generate_report():
    
    current_dir = os.path.dirname(os.path.abspath(__file__))  # 현재 파일(app.py)이 있는 디렉터리의 절대 경로
    final_src_path = os.path.join(current_dir, "final_src.py")
    if not os.path.exists(final_src_path):
        print(f"Error: final_src.py not found at {final_src_path}")
        return 
    
    data_src_path = os.path.join(current_dir, "data_src")
    output_folder = os.path.join(current_dir, "data_output")
    final_report_path = os.path.join(current_dir, "final_report.csv")
    response_file_path = os.path.join(data_src_path, "response_times.csv")

    try:
        subprocess.run(["python", final_src_path], check=True)

        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.cell(200, 10, txt="Survey Analysis Report", ln=True, align='C')
        pdf.ln(10)

        if os.path.exists(final_report_path) and os.path.exists(response_file_path):
            df = pd.read_csv(final_report_path)
            response_df = pd.read_csv(response_file_path)

            for index, row in df.iterrows():
                print(f"Processing question {index + 1} ...")
                qnum = int(row['qnum'])
                adjusted_response = int(row['eeg_num'])
                original_response = response_df[response_df['qnum'] == qnum]['response'].values[0]
                response_time = response_df[response_df['qnum'] == qnum]['res_time'].values[0]

            # 각 문항에 대한 정보 작성
                pdf.cell(200, 10, txt=f"Question {qnum}:", ln=True)
                pdf.cell(200, 10, txt=f"  - Original Response: {original_response}", ln=True)
                pdf.cell(200, 10, txt=f"  - Adjusted Response: {adjusted_response}", ln=True)
                pdf.cell(200, 10, txt=f"  - Response Time: {response_time:.2f} seconds", ln=True)
                pdf.ln(5)
                
                plot_file = os.path.join(output_folder, f"eeg_data_{qnum}_fft_confidence_plot.jpg")
                if os.path.exists(plot_file):
                    pdf.image(plot_file, x=10, y=pdf.get_y(), w=180)
                    pdf.ln(85)
                    print(f"Added graph for Question {qnum} to PEF.")
                else:
                    pdf.cell(200, 10, txt=f"No plot available for Question {qnum}.", ln=True)
                    pdf.ln(10)
        else :
            print("Error: final_report.csv or response_times.csv not found.")
            pdf.cell(200, 10, txt="Error: Analysis files not found.", ln=True, align='C')
    # PDF 저장
        pdf_output_path = "survey_report.pdf"
        pdf.output(pdf_output_path)
        print(f"survey analysis report saved to {pdf_output_path}")
    
    except subprocess.CalledProcessError as e:
        print(f"Error occurred while running final_src.py: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        if username in users and users[username] == password:
            session['user'] = username  # 세션에 사용자 저장
            return redirect(url_for('start_page'))  # 시작 페이지로 리다이렉트
        else:
            error = "ID나 password를 다시 확인해주세요"
            return render_template('login.html', error=error)

    return render_template('login.html')
    
@app.route('/', methods=['GET', 'POST'])
def start_page():
    if 'user' not in session:  # 세션에 사용자가 없으면 로그인 페이지로 리다이렉트
        return redirect(url_for('login'))
    
    if request.method == 'POST':
        question_number.value = 1
        start_data_collection(data_collection_active, eeg_data, gaze_data, question_number)
        return redirect(url_for('survey', question_number=1))
    return render_template('start.html')

@app.route('/logout')
def logout():
    session.pop('user', None)  # 세션에서 사용자 제거
    return redirect(url_for('login'))  # 로그인 페이지로 리다이렉트

@app.route('/survey', methods=['GET', 'POST'])
def survey():
    global question_number
    if request.method == 'POST':
        question_number.value = int(request.form['question_number'])
        response = request.form['response']

        # 응답시간 계산
        response_time = time.time() - float(request.form['start_time'])
        response_times.append([question_number.value, response, response_time])

        # 다음 or 제출
        if question_number.value < len(questions):
            question_number.value += 1
            return redirect(url_for('survey', question_number=question_number.value))
        else:
            stop_data_collection()
            save_data_to_csv(gaze_data, response_times)

            analysis_thread = threading.Thread(target=analyze_and_generate_report)
            analysis_thread.start()

            return redirect(url_for('loading_page'))

    question_number.value = int(request.args.get('question_number', 1))
    start_time = time.time()

    return render_template('survey.html', question_number=question_number.value, question=questions[question_number.value - 1], choices=choices, start_time=start_time)

@app.route('/loading', methods=['GET'])
def loading_page():
    if os.path.exists('survey_report.pdf'):
        return redirect(url_for('result'))
    else:
        return render_template('loading.html')

@app.route('/check_status', methods=['GET'])
def check_status():
    if os.path.exists('survey_report.pdf'):  # PDF 파일 생성 여부 확인
        return jsonify({'status': 'complete'})
    else:
        return jsonify({'status': 'processing'})


@app.route('/result', methods=['GET'])
def result() :
    
    response_times_path = os.path.join('data_src', 'response_times.csv')
    if not os.path.exists('final_report.csv'):
        return redirect(url_for('loading_page'))
    
    e_total, i_total = 0, 0
    s_total, n_total = 0, 0
    f_total, t_total = 0, 0
    p_total, j_total = 0, 0

    e_i_questions = [1, 3]
    s_n_questions = [2, 5]
    f_t_questions = [7, 8]
    p_j_questions = [4, 6]

    if os.path.exists(response_times_path):
        with open(response_times_path, newline='') as csvfile:
            reader = csv.reader(csvfile)
            next(reader) 
            for row in reader :
                question_number = int(row[0])
                response_value = int(row[1])

                if question_number in e_i_questions:
                    e_total += response_value
                elif question_number in s_n_questions:
                    s_total += response_value
                elif question_number in f_t_questions:
                    f_total += response_value
                elif question_number in p_j_questions:
                    p_total += response_value 
    
    e_per = round((e_total / 10) * 100, 1)
    i_per = round((10-e_total) / 10 * 100 ,1)
    s_per = round((s_total / 10) * 100, 1)
    n_per = round((10-s_total) / 10 * 100, 1)
    f_per = round((f_total / 10) * 100, 1)
    t_per = round((10-f_total) / 10 * 100, 1)
    p_per = round((p_total / 10) * 100, 1)
    j_per = round((10-p_total) / 10 * 100, 1)
    
    # 분석된 결과 데이터 로드 및 퍼센트 계산
    adjusted_results = {}
    if os.path.exists('final_report.csv'):
        df = pd.read_csv('final_report.csv')
        e_adjusted, i_adjusted = 0, 0
        s_adjusted, n_adjusted = 0, 0
        f_adjusted, t_adjusted = 0, 0
        p_adjusted, j_adjusted = 0, 0

        for _, row in df.iterrows():
            question_number = int(row['qnum'])
            response_value = int(row['eeg_num'])

            if question_number in e_i_questions:
                e_adjusted += response_value
            elif question_number in s_n_questions:
                s_adjusted += response_value
            elif question_number in f_t_questions:
                f_adjusted += response_value
            elif question_number in p_j_questions:
                p_adjusted += response_value

        e_adjusted_per = round((e_adjusted / 10) * 100, 1)
        i_adjusted_per = round((10 - e_adjusted) / 10 * 100, 1)
        s_adjusted_per = round((s_adjusted / 10) * 100, 1)
        n_adjusted_per = round((10 - s_adjusted) / 10 * 100, 1)
        f_adjusted_per = round((f_adjusted / 10) * 100, 1)
        t_adjusted_per = round((10 - f_adjusted) / 10 * 100, 1)
        p_adjusted_per = round((p_adjusted / 10) * 100, 1)
        j_adjusted_per = round((10 - p_adjusted) / 10 * 100, 1)

        adjusted_results = {
            'e_per': e_adjusted_per,
            'i_per': i_adjusted_per,
            's_per': s_adjusted_per,
            'n_per': n_adjusted_per,
            'f_per': f_adjusted_per,
            't_per': t_adjusted_per,
            'p_per': p_adjusted_per,
            'j_per': j_adjusted_per
        }

    return render_template('result.html',
                           e_per=e_per,
                           i_per=i_per,
                           s_per=s_per,
                           n_per=n_per,
                           f_per=f_per,
                           t_per=t_per,
                           p_per=p_per,
                           j_per=j_per,
                           adjusted_results=adjusted_results)

@app.route('/download_report', methods=['GET'])
def download_report():
    pdf_output_path = 'survey_report.pdf'
    if not os.path.exists(pdf_output_path):
        return "PDF가 아직 생성되지 않았습니다. 잠시 후 다시 시도해 주세요.", 404
    return send_file(pdf_output_path, as_attachment=True)


if __name__ == "__main__":
    manager = multiprocessing.Manager()
    data_collection_active = multiprocessing.Value('b', False)
    eeg_data = manager.list()
    gaze_data = manager.list()
    response_times = manager.list()
    question_number = manager.Value('i', 1)

    app.run(debug=True)
