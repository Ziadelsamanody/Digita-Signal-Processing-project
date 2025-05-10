import os 
import soundfile as sf
import librosa
import numpy as np
from pathlib import Path

class Voice_processor:
    
    @staticmethod
    def remove_noise(audio_path, output_path=None, noise_threshold=1000):
        try:
            y , sr_rate = librosa.load(audio_path)

            #SIFT analysis
            D = librosa.stft(y)
            magntuide = np.abs(D)
            phase = np.angle(D)

            # Noise Estmation  frequancy bins
            noise_level = np.mean(magntuide[:noise_threshold])

            #Specral subtraction
            cleaned_magntude =  np.maximum(magntuide - noise_level, 0)
            cleaned_D = cleaned_magntude * np.exp(1j * phase)

            cleaned_audio = librosa.istft(cleaned_D)

            # detrimaned output path
            if output_path is None:
                output_path = f"cleaned_{Path(audio_path).name}"

            #save results
            sf.write(output_path, cleaned_audio, sr_rate)
            return output_path
        
        except Exception as e :
            raise ValueError(f"Audio noise removal failed : {str(e)}")
        

        



