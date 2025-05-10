import os 
import numpy as np
import cv2
import librosa
from voises import Voice_processor
from images import ImageProcessor
import soundfile as sf
from tkinter import Tk, filedialog, messagebox, Button, Label, Frame
from tkinter import *


# Remove noise and save to a new file
def remove_noise(audio_path):
    y, sr_rate = librosa.load(audio_path)
    D = librosa.stft(y)
    magnitude = np.abs(D)
    noise_level = np.mean(magnitude[:1000])
    cleaned_magnitude = np.maximum(magnitude - noise_level, 0)
    cleaned_D = cleaned_magnitude * np.exp(1j * np.angle(D))
    cleaned_audio = librosa.istft(cleaned_D)
    cleaned_path = 'cleaned_audio.wav'
    sf.write(cleaned_path, cleaned_audio, sr_rate)





def process_selected_images():
    root = Tk()
    root.withdraw()

    file_paths = filedialog.askopenfilenames(
        title="Select image(s)",
        filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp")]
    )
    if not file_paths:
        print("No images selected.")
        return

    output_dir = filedialog.askdirectory(title="Select folder to save denoised images")
    if not output_dir:
        print("No output folder selected.")
        return

    print("\nStarting denoising process...\n")

    for i, path in enumerate(file_paths, start=1):
        filename = os.path.basename(path)
        output_path = os.path.join(output_dir, filename)

        try:
            print(f"[{i}/{len(file_paths)}] Processing: {filename}")
            processor = ImageProcessor(path=path)
            processor.denoise_image()
            processor.save_image(output_path)
            # processed_count += 1
          
        except Exception as e:
            print(f"Error processing {filename}: {e}")

    print("\nAll images have been processed and saved to:", output_dir)
    messagebox.showinfo("Process Complete", f"All images have been processed and saved to:\n{output_dir}")






def process_audio_file():
    root = Tk()
    root.withdraw()
    
    audio_path = filedialog.askopenfilename(
        title="Select audio file",
        filetypes=[("Audio files", "*.wav *.mp3 *.ogg *.flac")]
    )
    
    if not audio_path:
        return
    
    output_path = filedialog.asksaveasfilename(
        title="Save cleaned audio as",
        defaultextension=".wav",
        filetypes=[("WAV files", "*.wav")]
    )
    
    if not output_path:
        return
    audio_processor = Voice_processor() 
    cleaned_path = Voice_processor.remove_noise(audio_path=audio_path,output_path=output_path)
    if cleaned_path:
             messagebox.showinfo("Success", f"Audio processed and saved to:\n{cleaned_path}")

def show_selection_dialog():
    root = Tk()
    root.title("Processing Selection")
    root.geometry("300x150")
    
    frame = Frame(root, padx=20, pady=20)
    frame.pack(expand=True)
    
    Label(frame, text="Choose processing type:").pack(pady=10)
    
    Button(frame, text="Image Processing", command=lambda: [root.destroy(), process_selected_images()]).pack(fill='x', pady=5)
    Button(frame, text="Sound Processing", command=lambda: [root.destroy(), process_audio_file()]).pack(fill='x', pady=5)
    
    root.mainloop()    



def main():
    # remove_noise('jackhammer.wav')
    show_selection_dialog()

if __name__ == '__main__':
    main()
