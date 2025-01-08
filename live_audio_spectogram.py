import numpy as np
import cv2
import pyaudio
from tkinter import Tk, filedialog, Canvas, Frame
from tkinter import ttk
from scipy.io import wavfile

# Audio configuration
CHUNK = 1024  # Number of frames per buffer
RATE = 44100  # Sampling rate in Hz
WINDOW_WIDTH = 800
WINDOW_HEIGHT = 400

# Helper function to normalize data
def normalize(data, min_val=0, max_val=255):
    return np.interp(data, (data.min(), data.max()), (min_val, max_val))

def live_mic_visualization(root):
    root.destroy()
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16, channels=1, rate=RATE, input=True, frames_per_buffer=CHUNK)
    try:
        spectrogram = np.zeros((WINDOW_HEIGHT, WINDOW_WIDTH), dtype=np.uint8)
        while True:
            data = np.frombuffer(stream.read(CHUNK), dtype=np.int16)
            fft_data = np.abs(np.fft.rfft(data))
            fft_data = normalize(fft_data, 0, WINDOW_HEIGHT)

            spectrogram[:, :-1] = spectrogram[:, 1:]
            spectrogram[:, -1] = np.clip(fft_data[:WINDOW_HEIGHT], 0, WINDOW_HEIGHT - 1)

            spectrogram_flipped = np.flipud(spectrogram)
            spectrogram_colored = cv2.applyColorMap(spectrogram_flipped, cv2.COLORMAP_JET)
            cv2.imshow("Live Audio Spectrogram (Mic)", spectrogram_colored)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    except Exception as e:
        print(f"Error in mic visualization: {e}")
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()
        cv2.destroyAllWindows()
        main_menu()

def wav_file_visualization(root):
    root.destroy()
    file_path = filedialog.askopenfilename(filetypes=[("WAV files", "*.wav")])
    if not file_path:
        main_menu()
        return

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
            spectrogram[:, -1] = np.clip(fft_data[:WINDOW_HEIGHT], 0, WINDOW_HEIGHT - 1)

            spectrogram_flipped = np.flipud(spectrogram)
            spectrogram_colored = cv2.applyColorMap(spectrogram_flipped, cv2.COLORMAP_JET)
            cv2.imshow("Live Audio Spectrogram (WAV)", spectrogram_colored)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    except Exception as e:
        print(f"Error in WAV visualization: {e}")
    finally:
        cv2.destroyAllWindows()
        main_menu()

def convert_image_to_wav(root):
    root.destroy()
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.bmp;*.tiff")])
    if not file_path:
        main_menu()
        return

    output_wav = filedialog.asksaveasfilename(defaultextension=".wav", filetypes=[("WAV files", "*.wav")])
    if not output_wav:
        main_menu()
        return

    try:
        image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        height, width = image.shape

        # Normalize image to [-1, 1] range
        image_normalized = (image / 255.0) * 2 - 1

        # Generate audio data using IFFT
        audio = np.fft.irfft(image_normalized, axis=0).flatten()

        # Normalize audio to int16 range
        audio = (audio / np.max(np.abs(audio)) * 32767).astype(np.int16)

        # Save as WAV
        wavfile.write(output_wav, RATE, audio)
        print(f"WAV saved: {output_wav}")
    except Exception as e:
        print(f"Error converting image to WAV: {e}")
    finally:
        main_menu()

def convert_wav_to_image(root):
    root.destroy()
    file_path = filedialog.askopenfilename(filetypes=[("WAV files", "*.wav")])
    if not file_path:
        main_menu()
        return

    output_image = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("Image files", "*.png")])
    if not output_image:
        main_menu()
        return

    try:
        rate, audio = wavfile.read(file_path)
        audio = audio.astype(np.float32) / 32768  # Normalize to [-1, 1]

        # Reshape to match image dimensions
        audio = audio[:WINDOW_WIDTH * WINDOW_HEIGHT]  # Trim to fit dimensions
        audio_matrix = audio.reshape(WINDOW_WIDTH, WINDOW_HEIGHT).T

        # Apply FFT to get frequency spectrum
        spectrum = np.abs(np.fft.rfft(audio_matrix, axis=0))

        # Normalize spectrum to [0, 255]
        spectrum_normalized = (spectrum / np.max(spectrum) * 255).astype(np.uint8)

        # Save as image
        cv2.imwrite(output_image, spectrum_normalized)
        print(f"Image saved: {output_image}")
    except Exception as e:
        print(f"Error converting WAV to image: {e}")
    finally:
        main_menu()

def main_menu():
    root = Tk()
    root.title("Audio Spectrogram Viewer")
    root.geometry("500x600")
    root.configure(bg="#1e1e2f")

    frame = Frame(root, bg="#2c2c3e", highlightbackground="#ffffff", highlightthickness=2, relief="groove")
    frame.place(relx=0.5, rely=0.5, anchor="center", width=400, height=500)

    style = ttk.Style()
    style.theme_use("clam")
    style.configure("TButton", font=("Arial", 12), padding=10)
    style.configure("TLabel", background="#2c2c3e", foreground="#ffffff", font=("Arial", 14))

    ttk.Label(frame, text="Select an Option", background="#2c2c3e", font=("Arial", 14)).pack(pady=20)

    ttk.Button(frame, text="Use USB Microphone", command=lambda: live_mic_visualization(root)).pack(pady=10)
    ttk.Button(frame, text="Upload WAV File", command=lambda: wav_file_visualization(root)).pack(pady=10)
    ttk.Button(frame, text="Convert Image to WAV", command=lambda: convert_image_to_wav(root)).pack(pady=10)
    ttk.Button(frame, text="Convert WAV to Image", command=lambda: convert_wav_to_image(root)).pack(pady=10)
    ttk.Button(frame, text="Exit", command=root.quit).pack(pady=10)

    root.mainloop()

if __name__ == "__main__":
    main_menu()
