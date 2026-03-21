import numpy as np
import sounddevice as sd

class AudioDetector:
    def __init__(self, threshold_db=110):
        self.threshold_db = threshold_db
        self.is_detected = False
        self.current_db = -60
        self.current_peak = 0
        self.peak_hold = 0
        self.persistence_count = 0
        self.required_persistence = 15  # ~0.35s sustained — blocks speech spikes
        self.calibration_frames = 60    # ~2-3 seconds at callback rate
        self.avg_ambient_sum = 0
        self.db_offset = 100            # Will be adjusted during calibration
        self.is_calibrating = True

        self.samplerate = 44100
        for rate in [44100, 48000, 16000]:
            try:
                sd.check_input_settings(samplerate=rate, channels=1)
                self.samplerate = rate
                break
            except Exception:
                continue

        self.stream = sd.InputStream(
            callback=self._audio_callback,
            samplerate=self.samplerate,
            channels=1
        )
        self.stream.start()

    def _audio_callback(self, indata, frames, time, status):
        # Remove DC offset
        clean = indata - np.mean(indata)
        rms = np.sqrt(np.mean(clean ** 2))
        self.current_peak = np.max(np.abs(clean))
        self.peak_hold = max(self.peak_hold, self.current_peak)

        # Convert to dB (relative scale)
        raw_db = 20 * np.log10(rms + 1e-9) + self.db_offset

        if self.calibration_frames > 0:
            self.avg_ambient_sum += raw_db
            self.calibration_frames -= 1
            if self.calibration_frames == 0:
                avg_ambient = self.avg_ambient_sum / 60
                # Normalize so a quiet room = 45 dB
                self.db_offset += (45 - avg_ambient)
                self.is_calibrating = False
                print(f"[AstraNaari] Room calibrated. Quiet room ≈ 45 dB. Ready.")
            return

        self.current_db = 20 * np.log10(rms + 1e-9) + self.db_offset

        # Persistence gate — must stay loud for 0.35s before triggering
        if self.current_db >= self.threshold_db:
            self.persistence_count = min(self.persistence_count + 1, 20)
        else:
            self.persistence_count = max(self.persistence_count - 2, 0)  # fast decay

        self.is_detected = self.persistence_count >= self.required_persistence

    def get_levels(self):
        peak = self.peak_hold
        self.peak_hold = self.current_peak
        return self.is_detected, self.current_db, peak

    def stop(self):
        self.stream.stop()
        self.stream.close()
