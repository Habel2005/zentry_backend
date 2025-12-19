import webrtcvad
import audioop

class VADStreamer:
    def __init__(self, sample_rate=16000, mode=3, min_energy=500):
        self.vad = webrtcvad.Vad(mode)
        self.sample_rate = sample_rate
        self.buffer = bytearray()
        self.speech_buffer = bytearray()
        self.in_speech = False
        self.silence_duration = 0
        self.min_energy = min_energy  # Minimum volume to consider "speech"

    def process_chunk(self, chunk):
        """
        Returns:
            - bytes: If a complete sentence is finished.
            - "BARGE_IN": If the user JUST started talking (interrupt trigger).
            - None: If nothing interesting happened.
        """
        self.buffer.extend(chunk)
        frame_size = 960  # 30ms @ 16kHz
        
        detected_utterance = None
        barge_in_triggered = False

        while len(self.buffer) >= frame_size:
            frame = self.buffer[:frame_size]
            self.buffer = self.buffer[frame_size:]

            # 1. Safety: Check Volume (Energy) before VAD
            # This prevents 4G "hiss" from triggering the bot
            energy = audioop.rms(frame, 2)
            is_speech = False
            
            if energy > self.min_energy:
                try:
                    is_speech = self.vad.is_speech(frame, self.sample_rate)
                except Exception:
                    pass # Ignore VAD errors on bad frames

            # 2. State Machine
            if is_speech:
                if not self.in_speech:
                    self.in_speech = True
                    barge_in_triggered = True  # <--- CRITICAL: Signal to kill TTS
                    self.speech_buffer = bytearray()
                self.speech_buffer.extend(frame)
                self.silence_duration = 0
            else:
                if self.in_speech:
                    self.speech_buffer.extend(frame)
                    self.silence_duration += 30
                    # 3. Tuning: 500ms is the "Sweet Spot" for human pauses vs stops
                    if self.silence_duration > 500: 
                        detected_utterance = bytes(self.speech_buffer)
                        self.in_speech = False
                        self.speech_buffer = bytearray()
                        self.silence_duration = 0

        if barge_in_triggered:
            return "BARGE_IN"
        
        return detected_utterance