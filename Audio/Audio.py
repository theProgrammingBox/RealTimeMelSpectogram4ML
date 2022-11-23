import pyaudio
import librosa
import numpy

ViewFirstFraction = 2	#1 for 100%, 2 for 50%, 100 for 1%
WindowSize = 203        #fullscreen is about 203 chars wide
MaxDB = 30              #play around with it, 30 - 80 is good based on your volume

p = pyaudio.PyAudio()

FORMAT = pyaudio.paFloat32
CHANNELS = 1
RATE = 44100
FRAMES_PER_BUFFER = 1024

stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=FRAMES_PER_BUFFER)

N_FFT = 4096
N_MELS = WindowSize * ViewFirstFraction
M = librosa.filters.mel(RATE, N_FFT, N_MELS)

while True:
    data = stream.read(FRAMES_PER_BUFFER)
    audio_data = numpy.fromstring(data, dtype=numpy.float32)
    spectrogram = librosa.feature.melspectrogram(y=audio_data, sr=RATE, n_fft=N_FFT, hop_length=FRAMES_PER_BUFFER + 1, n_mels=N_MELS)
    melSpectrogram = librosa.power_to_db(spectrogram, ref=numpy.max)
    
    illuminationList = ".,-~:;=!*#$@"
    melSpectrogram = melSpectrogram + MaxDB
    melSpectrogram[melSpectrogram < 0] = 0
    melSpectrogram[melSpectrogram > MaxDB - 0.001] = MaxDB - 0.001
    melSpectrogram = melSpectrogram / MaxDB * len(illuminationList)
    string = ""
    for i in range(0, WindowSize):
        string += illuminationList[int(melSpectrogram[i])]
    print(string)