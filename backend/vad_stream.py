import numpy as np
import logging
import os

class VADStreamer:
    def __init__(self, sample_rate=8000, threshold=0.6, min_energy=0.015): 
        # [TUNING] 
        # min_energy=0.015: This is the fix for "Roed..."/"Thiom..." hallucinations.
        # It forces the VAD to IGNORE quiet phone line static.
        
        self.sample_rate = sample_rate
        self.threshold = threshold
        self.min_energy = min_energy

        self.window_size_samples = 256 if sample_rate == 8000 else 512
        self.buffer = bytearray()
        self.speech_buffer = bytearray()
        self.in_speech = False
        self.silence_duration = 0
        
        # 0.8s silence to mark end of sentence
        self.max_silence_chunks = int(25) 
        
        # Max 15s of speech
        self.max_speech_buffer = 15 * sample_rate * 2 

        self.session = None
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._sr = np.array([sample_rate], dtype=np.int64)
        
        self.load_model()
        self.debug_counter = 0

    def load_model(self):
        try:
            import onnxruntime
            options = onnxruntime.SessionOptions()
            options.log_severity_level = 3
            self.session = onnxruntime.InferenceSession(
                "models/silero_vad.onnx", options, providers=["CPUExecutionProvider"]
            )
            print(f"✅ VAD Loaded (SR: {self.sample_rate})")
        except Exception as e:
            print(f"❌ VAD Load Failed: {e}")

    def process_chunk(self, chunk):
        self.buffer.extend(chunk)
        required_bytes = self.window_size_samples * 2
        
        detected_utterance = None

        while len(self.buffer) >= required_bytes:
            frame_bytes = self.buffer[:required_bytes]
            self.buffer = self.buffer[required_bytes:]
            
            # Convert to Float32
            input_tensor = np.frombuffer(frame_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            
            # --- ENERGY GATE ---
            energy = np.abs(input_tensor).mean()
            
            # Debug: Only print if energy is somewhat significant
            self.debug_counter += 1
            if self.debug_counter % 50 == 0 and energy > 0.005:
                print(f"[VAD] Energy: {energy:.4f}   ", end="\r")

            input_tensor = input_tensor.reshape(1, -1)
            
            speech_prob = 0.0
            if self.session:
                try:
                    ort_inputs = {"input": input_tensor, "sr": self._sr, "state": self._state}
                    ort_outs = self.session.run(None, ort_inputs)
                    speech_prob = ort_outs[0][0][0]
                    self._state = ort_outs[1]
                except Exception:
                    pass
            
            # [STRICT TRIGGER]
            # Must be PROBABLE speech AND LOUD enough
            is_speech = (speech_prob > self.threshold) and (energy > self.min_energy)

            if is_speech:
                if not self.in_speech:
                    self.in_speech = True
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
                        print(f"\n[VAD] 🗣️ Captured {len(detected_utterance)} bytes")

            # Safety Cutoff
            if self.in_speech and len(self.speech_buffer) > self.max_speech_buffer:
                detected_utterance = bytes(self.speech_buffer)
                self.in_speech = False
                self.speech_buffer = bytearray()
                self._state = np.zeros((2, 1, 128), dtype=np.float32)

        return detected_utterance