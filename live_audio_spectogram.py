import numpy as np
import cv2
import pyaudio
from tkinter import Tk, filedialog, Button, Label, Canvas, Frame
from tkinter import ttk
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
    root = Tk()
    root.title("Live Mic Visualization")
    root.geometry("500x500")
    root.configure(bg="#1e1e2f")

    back_button = ttk.Button(root, text="Back", command=lambda: [root.destroy(), main_menu()])
    back_button.pack(pady=20)

    label = ttk.Label(root, text="Live visualization in progress... Close this window to return.", font=("Arial", 12), background="#1e1e2f", foreground="#ffffff")
    label.pack(pady=20)

    root.mainloop()

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
            spectrogram[:, -1] = np.clip(fft_data[:WINDOW_HEIGHT], 0, WINDOW_HEIGHT - 1)

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
    root = Tk()
    root.title("WAV File Visualization")
    root.geometry("500x500")
    root.configure(bg="#1e1e2f")

    back_button = ttk.Button(root, text="Back", command=lambda: [root.destroy(), main_menu()])
    back_button.pack(pady=20)

    label = ttk.Label(root, text="Visualization in progress... Close this window to return.", font=("Arial", 12), background="#1e1e2f", foreground="#ffffff")
    label.pack(pady=20)

    root.mainloop()

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
            cv2.imshow("Live Audio Spectrogram (WAV)", spectrogram_flipped)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    except KeyboardInterrupt:
        print("WAV visualization stopped.")
    finally:
        cv2.destroyAllWindows()

def image_to_wav(image_path, output_wav, sample_rate=44100):
    """Convert an image to a WAV file."""
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    height, width = image.shape

    # Normalize image to [-1, 1] range
    image_normalized = (image / 255.0) * 2 - 1

    # Generate audio data using IFFT
    audio = np.fft.irfft(image_normalized, axis=0).flatten()

    # Normalize audio to int16 range
    audio = (audio / np.max(np.abs(audio)) * 32767).astype(np.int16)

    # Save as WAV
    wavfile.write(output_wav, sample_rate, audio)
    print(f"WAV saved: {output_wav}")

def wav_to_image(wav_path, output_image, image_height, image_width):
    """Convert a WAV file to an image."""
    rate, audio = wavfile.read(wav_path)
    audio = audio.astype(np.float32) / 32768  # Normalize to [-1, 1]

    # Reshape to match image dimensions
    audio = audio[:image_width * image_height]  # Trim to fit dimensions
    audio_matrix = audio.reshape(image_width, image_height).T

    # Apply FFT to get frequency spectrum
    spectrum = np.abs(np.fft.rfft(audio_matrix, axis=0))

    # Normalize spectrum to [0, 255]
    spectrum_normalized = (spectrum / np.max(spectrum) * 255).astype(np.uint8)

    # Save as image
    cv2.imwrite(output_image, spectrum_normalized)
    print(f"Image saved: {output_image}")

def convert_image_to_wav():
    """Select an image and convert it to a WAV file."""
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.bmp;*.tiff")])
    if file_path:
        output_wav = filedialog.asksaveasfilename(defaultextension=".wav", filetypes=[("WAV files", "*.wav")])
        if output_wav:
            image_to_wav(file_path, output_wav)

def convert_wav_to_image():
    """Select a WAV file and convert it to an image."""
    file_path = filedialog.askopenfilename(filetypes=[("WAV files", "*.wav")])
    if file_path:
        output_image = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("Image files", "*.png")])
        if output_image:
            wav_to_image(file_path, output_image, image_height=WINDOW_HEIGHT, image_width=WINDOW_WIDTH)

def main_menu():
    """Create a modern GUI for user selection."""
    root = Tk()
    root.title("Audio Spectrogram Viewer")
    root.geometry("500x600")
    root.configure(bg="#1e1e2f")

    # Create a rounded canvas for modern look
    canvas = Canvas(root, bg="#1e1e2f", bd=0, highlightthickness=0)
    canvas.pack(fill="both", expand=True)

    frame = Frame(canvas, bg="#2c2c3e", highlightbackground="#ffffff", highlightthickness=2, relief="groove")
    frame.place(relx=0.5, rely=0.5, anchor="center", width=400, height=500)

    style = ttk.Style()
    style.theme_use("clam")
    style.configure("TButton", font=("Arial", 12), padding=10, background="#4caf50", foreground="white")
    style.configure("TLabel", background="#2c2c3e", foreground="#ffffff", font=("Arial", 14))

    label = ttk.Label(frame, text="Select an Option")
    label.pack(pady=20)

    mic_button = ttk.Button(frame, text="Use USB Microphone", command=lambda: [root.destroy(), live_mic_visualization()])
    mic_button.pack(pady=10)

    wav_button = ttk.Button(frame, text="Upload WAV File", command=lambda: [root.destroy(), convert_wav_to_image()])
    wav_button.pack(pady=10)

    image_to_wav_button
