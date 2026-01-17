import numpy as np
import logging
import os

class VADStreamer:
    def __init__(self, sample_rate=8000, threshold=0.4): # [FIX] Lowered to 0.4
        self.sample_rate = sample_rate
        self.threshold = threshold
        
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
        
        # Wait 0.8s of silence before marking sentence as done
        self.max_silence_chunks = int(800 / 32) 
        
        # Max 6s of speech to prevent infinite buffering
        self.max_speech_buffer = 6 * 16000 * 2 

        self.session = None
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._sr = np.array([sample_rate], dtype=np.int64)
        
        self.load_model()
        self.debug_counter = 0

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

    def process_chunk(self, chunk):
        self.buffer.extend(chunk)
        required_bytes = self.window_size_samples * 2
        
        detected_utterance = None
        barge_in_triggered = False

        while len(self.buffer) >= required_bytes:
            frame_bytes = self.buffer[:required_bytes]
            self.buffer = self.buffer[required_bytes:]
            
            input_tensor = np.frombuffer(frame_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            
            # [DEBUG] Print energy levels occasionally
            self.debug_counter += 1
            if self.debug_counter % 50 == 0:
                energy = np.abs(input_tensor).mean()
                if energy > 0.01: # Only print if there is actual sound
                    print(f"[VAD] Signal Energy: {energy:.4f}", end="\r")

            input_tensor = input_tensor.reshape(1, -1)
            
            speech_prob = 0.0
            if self.session:
                try:
                    ort_inputs = {
                        "input": input_tensor,
                        "sr": self._sr,
                        "state": self._state
                    }
                    ort_outs = self.session.run(None, ort_inputs)
                    speech_prob = ort_outs[0][0][0]
                    self._state = ort_outs[1]
                except Exception:
                    pass
            
            if speech_prob > self.threshold:
                if self.debug_counter % 10 == 0:
                     print(f"[VAD] 🗣️ Speech Detected! (Prob: {speech_prob:.2f})", end="\r")

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
                        print("\n[VAD] 🤫 Silence detected. Processing utterance...")

            # Safety Cutoff
            if self.in_speech and len(self.speech_buffer) > self.max_speech_buffer:
                logging.info("\n✂️ Force-cutting long speech segment")
                detected_utterance = bytes(self.speech_buffer)
                self.in_speech = False
                self.speech_buffer = bytearray()
                self.silence_duration = 0
                self._state = np.zeros((2, 1, 128), dtype=np.float32)

        if barge_in_triggered:
            return "BARGE_IN"
        
        return detected_utterance