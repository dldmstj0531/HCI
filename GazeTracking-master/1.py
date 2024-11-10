import threading
from gaze_tracking import GazeTracking
from muselsl import stream, list_muses, record
from example import gaze_track
from muse_example import muse_EEG

def Record_planner():
    # gaze_track을 별도 스레드에서 실행
    gaze_thread = threading.Thread(target=gaze_track)
    
    # 스레드 시작
    gaze_thread.start()
    
    # muse_EEG는 메인 스레드에서 실행
    muse_EEG()
    
    # gaze_track 스레드가 완료될 때까지 대기
    gaze_thread.join()

if __name__ == "__main__":
    Record_planner()
