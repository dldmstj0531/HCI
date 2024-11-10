import threading
import time
import tkinter as tk
import pandas as pd
import cv2  # cv2 모듈 추가
from tkinter import filedialog, StringVar
from gaze_tracking import GazeTracking
from example import gaze_track

class GazeApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Gaze Tracking App")
        self.root.geometry("600x400")
        self.start_time = None
        self.csv_file = None
        self.questions_df = None
        self.current_question_index = -1
        self.gaze_data = []  # 시선 추적 데이터를 저장할 리스트
        
        self.file_button = tk.Button(self.root, text="CSV 파일 찾기", font=("Arial", 12), command=self.select_file)
        self.file_button.pack(pady=10)
        
        self.file_label = tk.Label(self.root, text="선택한 파일 없음", font=("Arial", 10))
        self.file_label.pack(pady=5)

        self.button = tk.Button(self.root, text="연결하기", command=self.connect_gaze_tracking, font=("Arial", 16), state="disabled")
        self.button.pack(pady=20)
        
        self.timer_label = tk.Label(self.root, text="", font=("Arial", 14))
        self.timer_label.pack(pady=10)
    
    def select_file(self):
        self.csv_file = filedialog.askopenfilename(
            title="파일 선택창",
            filetypes=(("CSV files", "*.csv"), ("all files", "*.*"))
        )
        if self.csv_file:
            self.file_label.config(text=f"선택한 파일명: {self.csv_file.split('/')[-1]}")
            self.button.config(state="normal")
        else:
            self.file_label.config(text="선택한 파일 없음")
            self.button.config(state="disabled")
    
    def connect_gaze_tracking(self):
        self.button.config(state="disabled")
        self.gaze_thread = threading.Thread(target=self.start_gaze_tracking)
        self.gaze_thread.start()
    
    def start_gaze_tracking(self):
        self.root.after(0, self.update_button_to_start)
        gaze_track()
    
    def update_button_to_start(self):
        self.button.config(text="시작하기", state="normal", command=self.start_quiz)
    
    def start_quiz(self):
        if self.csv_file:
            self.questions_df = pd.read_csv(self.csv_file)
            self.start_time = time.time()  # 타이머 시작
            self.track_gaze()  # 시선 추적 시작
            self.next_question()
    
    def track_gaze(self):
        gaze = GazeTracking()
        webcam = cv2.VideoCapture(0)

        def record_gaze_data():
            while True:
                _, frame = webcam.read()
                gaze.refresh(frame)

                # 시선 상태 판별
                if gaze.is_blinking():
                    gaze_status = "Blinking"
                elif gaze.is_right():
                    gaze_status = "Looking right"
                elif gaze.is_left():
                    gaze_status = "Looking left"
                elif gaze.is_center():
                    gaze_status = "Looking center"
                else:
                    gaze_status = "Undetected"

                elapsed_time = time.time() - self.start_time
                # 현재 문제 번호와 함께 시선 데이터를 저장
                self.gaze_data.append({
                    'Time': elapsed_time,
                    'Gaze': gaze_status,
                    'Question': self.current_question_index + 1  # 현재 문제 번호
                })

                time.sleep(1)  # 1초 간격으로 측정

        # 스레드로 시선 추적 시작
        threading.Thread(target=record_gaze_data, daemon=True).start()
    
    def show_question(self):
        for widget in self.root.winfo_children():
            widget.destroy()

        question_number = self.questions_df.iloc[self.current_question_index, 0]
        question_text = self.questions_df.iloc[self.current_question_index, 1]

        question_label = tk.Label(self.root, text=f"문제 {question_number}: {question_text}", font=("Arial", 14))
        question_label.pack(pady=30)
        
        self.selected_option = StringVar()
        self.selected_option.set(None)

        options = ["전혀 그렇지 않다", "그렇지 않다", "보통이다", "그렇다", "매우 그렇다"]
        
        # 첫 번째 줄과 두 번째 줄의 선택지 위치 설정
        positions = {
            "전혀 그렇지 않다": (50, 100),  # 첫 번째 줄 왼쪽
            "보통이다": (250, 100),         # 첫 번째 줄 가운데
            "매우 그렇다": (450, 100),      # 첫 번째 줄 오른쪽
            "그렇지 않다": (150, 150),      # 두 번째 줄 왼쪽 중간
            "그렇다": (350, 150)            # 두 번째 줄 오른쪽 중간
        }

        # 라디오 버튼 생성 및 배치
        for option in options:
            x, y = positions[option]
            radio_button = tk.Radiobutton(
                self.root, text=option, variable=self.selected_option, value=option, font=("Arial", 12)
            )
            radio_button.place(x=x, y=y)

        next_button = tk.Button(self.root, text="다음", command=self.next_question, font=("Arial", 12))
        next_button.place(x=250, y=250)
        
    def next_question(self):
        self.current_question_index += 1

        if self.current_question_index < len(self.questions_df):
            self.show_question()
        else:
            self.show_completion_message()

    def show_completion_message(self):
        for widget in self.root.winfo_children():
            widget.destroy()
        completion_label = tk.Label(self.root, text="모든 문제를 완료했습니다!", font=("Arial", 16))
        completion_label.pack(pady=20)

        # CSV 파일로 시선 데이터 저장
        gaze_df = pd.DataFrame(self.gaze_data)
        save_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")])
        if save_path:
            gaze_df.to_csv(save_path, index=False)
            tk.Label(self.root, text="시선 데이터가 저장되었습니다.", font=("Arial", 12)).pack(pady=10)

if __name__ == "__main__":
    root = tk.Tk()
    app = GazeApp(root)
    root.mainloop()
