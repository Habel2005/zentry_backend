import numpy as np
import logging
import os

class VADStreamer:
    def __init__(self, sample_rate=8000, threshold=0.5):
        self.sample_rate = sample_rate
        self.threshold = threshold
        
        # Silero VAD v5 window sizes: 256, 512, 768 for 8k/16k 
        if sample_rate == 16000:
            self.window_size_samples = 512
        elif sample_rate == 8000:
            self.window_size_samples = 256
        else:
            raise ValueError("Silero VAD only supports 8000 or 16000 Hz")

        self.buffer = bytearray()
        self.speech_buffer = bytearray()
        self.in_speech = False
        self.silence_duration = 0
        self.max_silence_chunks = int(1000 / 32)

        # [FIX] v5 uses a single 'state' tensor instead of 'h' and 'c' 
        self.session = None
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._sr = np.array([sample_rate], dtype=np.int64)
        
        self.load_model()

    def load_model(self):
        try:
            import onnxruntime
            model_path = "models/silero_vad.onnx"
            options = onnxruntime.SessionOptions()
            options.log_severity_level = 3
            self.session = onnxruntime.InferenceSession(
                model_path, options, providers=["CPUExecutionProvider"]
            )
            logging.info(f"✅ Silero VAD v5 Loaded (SR: {self.sample_rate})")
        except Exception as e:
            logging.error(f"❌ Failed to load Silero VAD: {e}")

    def reset_states(self):
        # Change this:
        # self._state = np.zeros((2, 1, 64), dtype=np.float32)

        # To this:
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self.in_speech = False
        self.speech_buffer = bytearray()

    def process_chunk(self, chunk):
        self.buffer.extend(chunk)
        required_bytes = self.window_size_samples * 2
        
        detected_utterance = None
        barge_in_triggered = False

        while len(self.buffer) >= required_bytes:
            frame_bytes = self.buffer[:required_bytes]
            self.buffer = self.buffer[required_bytes:]
            
            input_tensor = np.frombuffer(frame_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            input_tensor = input_tensor.reshape(1, -1)

            speech_prob = 0.0
            if self.session:
                try:
                    # [FIX] Update input feed to use 'state' instead of 'h' and 'c'
                    ort_inputs = {
                        "input": input_tensor,
                        "sr": self._sr,
                        "state": self._state
                    }
                    ort_outs = self.session.run(None, ort_inputs)
                    speech_prob = ort_outs[0][0][0]
                    # [FIX] Update the internal state with the new state from output
                    self._state = ort_outs[1] 
                except Exception as e:
                    logging.error(f"VAD Inference Error: {e}")
            
            if speech_prob > self.threshold:
                if not self.in_speech:
                    self.in_speech = True
                    barge_in_triggered = True
                    self.speech_buffer = bytearray()
                self.speech_buffer.extend(frame_bytes)
                self.silence_duration = 0
            else:
                if self.in_speech:
                    self.speech_buffer.extend(frame_bytes)
                    self.silence_duration += 1
                    if self.silence_duration > self.max_silence_chunks:
                        detected_utterance = bytes(self.speech_buffer)
                        self.in_speech = False
                        self.speech_buffer = bytearray()
                        self.silence_duration = 0
        
        if barge_in_triggered:
            return "BARGE_IN"
        
        return detected_utterance