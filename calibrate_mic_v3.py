import sounddevice as sd
import numpy as np
import time

def callback(indata, frames, time, status):
    clean = indata - np.mean(indata)
    rms = np.abs(np.mean(clean**2))**0.5
    peak = np.max(np.abs(clean))
    zcr = len(np.where(np.diff(np.sign(clean)))[0]) / len(clean)
    print(f"Peak: {peak:.3f} | RMS: {rms:.3f} | ZCR: {zcr:.3f}")

print("--- MIC DIAGNOSTIC TOOL v3 ---")
print("1. Please stay SILENT for 3 seconds...")
time.sleep(3)
print("2. PLEASE SHOUT 'HELP!' NOW! (Recording for 5 seconds)")

with sd.InputStream(callback=callback, samplerate=44100, channels=1):
    time.sleep(5)

print("\nDONE. Please copy-paste the last 10 lines of output above!")
