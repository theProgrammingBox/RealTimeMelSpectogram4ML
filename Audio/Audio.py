import pyaudio
import wave
import librosa
import numpy

ViewFirstFraction = 2	#1 for 100%, 2 for 50%, 100 for 1%
WindowSize = 203

FORMAT = pyaudio.paFloat32
CHANNELS = 1
RATE = 44100
FRAMES_PER_BUFFER = 1024

p = pyaudio.PyAudio()

N_FFT = 4096
N_MELS = WindowSize * ViewFirstFraction
M = librosa.filters.mel(RATE, N_FFT, N_MELS)

stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=FRAMES_PER_BUFFER)

while True:
    data = stream.read(FRAMES_PER_BUFFER)
    audio_data = numpy.fromstring(data, dtype=numpy.float32)
    spectrogram = librosa.feature.melspectrogram(y=audio_data, sr=RATE, n_fft=N_FFT, hop_length=FRAMES_PER_BUFFER + 1, n_mels=N_MELS)
    melSpectrogram = librosa.power_to_db(spectrogram, ref=numpy.max)
    
    illuminationList = ".,-~:;=!*#$@"
    melSpectrogram = melSpectrogram + 80
    melSpectrogram[melSpectrogram < 0] = 0
    melSpectrogram[melSpectrogram > 79.999] = 79.999
    melSpectrogram = melSpectrogram / 80 * len(illuminationList)
    string = ""
    for i in range(0, WindowSize):
        string += illuminationList[int(melSpectrogram[i])]
    print(string)