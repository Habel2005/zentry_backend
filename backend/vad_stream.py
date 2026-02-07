import numpy as np
import logging
import os

class VADStreamer:
    def __init__(self, sample_rate=16000, threshold=0.4, min_energy=0.015): 
        self.sample_rate = sample_rate
        self.threshold = threshold
        self.min_energy = min_energy
        
        # [TUNING] Energy Override
        # If energy is above this, we IGNORE the AI probability and force trigger.
        # 0.1 is safe for loud speaking (your logs showed 0.17).
        self.energy_override = 0.08 

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
        # Silero V4 uses (2, 1, 128) state
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._sr = np.array([sample_rate], dtype=np.int64)
        
        self.load_model()
        self.debug_counter = 0

    def load_model(self):
        try:
            import onnxruntime
            options = onnxruntime.SessionOptions()
            options.log_severity_level = 3
            # Ensure we use the V4 model path you downloaded
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
            
            # Debug Logs
            self.debug_counter += 1
            if self.debug_counter % 20 == 0: 
                if energy > 0.005:
                    status = "🔴 REC" if self.in_speech else "⚪ WAITING"
                    # Log both metrics
                    print(f"[VAD] {status} | Energy: {energy:.4f} | Prob: {speech_prob:.4f}", end="\r")

            # [HYBRID TRIGGER LOGIC]
            # 1. Standard AI: Prob > Threshold AND Energy > Min
            ai_trigger = (speech_prob > self.threshold) and (energy > self.min_energy)
            
            # 2. Force Override: Energy > Override (0.08)
            # This fixes the "Upsampling Ghost" issue where Prob is 0.0005 but Energy is 0.17
            force_trigger = energy > self.energy_override

            is_speech = ai_trigger or force_trigger

            if is_speech:
                if not self.in_speech:
                    print(f"\n[VAD] 🗣️ Speech Started (Trigger: {'FORCE' if force_trigger and not ai_trigger else 'AI'})")
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
                        # Reset state to prevent "confused" context from sticking around
                        self._state = np.zeros((2, 1, 128), dtype=np.float32)
                        print(f"\n[VAD] 📦 Captured {len(detected_utterance)} bytes")

            # Safety Cutoff (15s max)
            if self.in_speech and len(self.speech_buffer) > self.max_speech_buffer:
                detected_utterance = bytes(self.speech_buffer)
                self.in_speech = False
                self.speech_buffer = bytearray()
                self._state = np.zeros((2, 1, 128), dtype=np.float32)

        return detected_utterance