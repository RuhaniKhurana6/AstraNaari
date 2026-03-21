import winsound
import threading

_is_playing = False

def play_buzzer(frequency=1500, duration=500):
    """Plays an asynchronous beep sound using a background thread and winsound.
    
    Args:
        frequency (int): Pitch of the buzzer in Hz (e.g., 1000 to 2500).
        duration (int): Duration of the buzzer in milliseconds.
    """
    global _is_playing
    if _is_playing:
        return
    
    _is_playing = True

    def beep_thread():
        global _is_playing
        try:
            winsound.Beep(frequency, duration)
        finally:
            _is_playing = False
            
    threading.Thread(target=beep_thread, daemon=True).start()
