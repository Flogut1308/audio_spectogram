import numpy as np
import cv2
import pyaudio
from tkinter import Tk, filedialog, Button, Label
from scipy.io import wavfile

# Audio configuration
CHUNK = 1024  # Number of frames per buffer
RATE = 44100  # Sampling rate in Hz
WINDOW_WIDTH = 800
WINDOW_HEIGHT = 400

def normalize(data, min_val=0, max_val=255):
    """Normalize data to a given range."""
    return np.interp(data, (data.min(), data.max()), (min_val, max_val))

def live_mic_visualization():
    """Visualize audio from a USB microphone in real-time."""
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16,
                    channels=1,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)
    try:
        spectrogram = np.zeros((WINDOW_HEIGHT, WINDOW_WIDTH), dtype=np.uint8)
        while True:
            data = np.frombuffer(stream.read(CHUNK), dtype=np.int16)
            fft_data = np.abs(np.fft.rfft(data))
            fft_data = normalize(fft_data, 0, WINDOW_HEIGHT)

            spectrogram[:, :-1] = spectrogram[:, 1:]
            spectrogram[:, -1] = np.clip(fft_data, 0, WINDOW_HEIGHT - 1)

            spectrogram_flipped = np.flipud(spectrogram)
            cv2.imshow("Live Audio Spectrogram (Mic)", spectrogram_flipped)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    except KeyboardInterrupt:
        print("Mic visualization stopped.")
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()
        cv2.destroyAllWindows()

def wav_file_visualization(file_path):
    """Visualize audio from a WAV file."""
    rate, data = wavfile.read(file_path)
    spectrogram = np.zeros((WINDOW_HEIGHT, WINDOW_WIDTH), dtype=np.uint8)
    start = 0

    try:
        while True:
            end = start + CHUNK
            if end > len(data):
                start = 0  # Loop the file
                end = start + CHUNK

            chunk_data = data[start:end]
            start += CHUNK

            fft_data = np.abs(np.fft.rfft(chunk_data))
            fft_data = normalize(fft_data, 0, WINDOW_HEIGHT)

            spectrogram[:, :-1] = spectrogram[:, 1:]
            spectrogram[:, -1] = np.clip(fft_data, 0, WINDOW_HEIGHT - 1)

            spectrogram_flipped = np.flipud(spectrogram)
            cv2.imshow("Live Audio Spectrogram (WAV)", spectrogram_flipped)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    except KeyboardInterrupt:
        print("WAV visualization stopped.")
    finally:
        cv2.destroyAllWindows()

def select_wav_file():
    """Open a file dialog to select a WAV file."""
    file_path = filedialog.askopenfilename(filetypes=[("WAV files", "*.wav")])
    if file_path:
        wav_file_visualization(file_path)

def main_menu():
    """Create a simple GUI for user selection."""
    root = Tk()
    root.title("Audio Spectrogram Viewer")

    label = Label(root, text="Select Input Source", font=("Arial", 14))
    label.pack(pady=20)

    mic_button = Button(root, text="Use USB Microphone", command=lambda: [root.destroy(), live_mic_visualization()])
    mic_button.pack(pady=10)

    wav_button = Button(root, text="Upload WAV File", command=lambda: [root.destroy(), select_wav_file()])
    wav_button.pack(pady=10)

    exit_button = Button(root, text="Exit", command=root.quit)
    exit_button.pack(pady=10)

    root.mainloop()

if __name__ == "__main__":
    main_menu()
