# muse_example.py
from muselsl import stream, list_muses, record

def muse_EEG():
    # Muse 장치 검색
    muses = list_muses()

    if not muses:
        print('No Muses found')
    else:
        stream(muses[0]['address'])

        # Note: Streaming is synchronous, so code here will not execute until the stream has been closed
        print('Stream has ended')

if __name__ == "__main__":
    muse_EEG()

