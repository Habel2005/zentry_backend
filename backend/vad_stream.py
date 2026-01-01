import numpy as np
import logging
import os

class VADStreamer:
    """
    VADStreamer using Silero VAD (ONNX) for high-performance voice activity detection.
    Supports 8000Hz and 16000Hz.
    """
    def __init__(self, sample_rate=16000, min_energy=0.1, threshold=0.5):
        self.sample_rate = sample_rate
        self.threshold = threshold
        self.min_energy = min_energy
        
        # Silero VAD Parameters
        # It works best with window sizes of 512, 1024, 1536 samples for 16000Hz.
        # For 8000Hz, we scale accordingly (256, 512, 768).
        if sample_rate == 16000:
            self.window_size_samples = 512 # 32ms
        elif sample_rate == 8000:
            self.window_size_samples = 256 # 32ms
        else:
            raise ValueError("Silero VAD only supports 8000 or 16000 Hz")

        self.buffer = bytearray()
        self.speech_buffer = bytearray()
        self.in_speech = False
        self.silence_duration = 0 # in chunks
        self.max_silence_chunks = int(500 / 32) # ~500ms of silence to stop

        # AI State for Silero
        self.session = None
        self._h = np.zeros((2, 1, 64), dtype=np.float32)
        self._c = np.zeros((2, 1, 64), dtype=np.float32)
        self._last_sr = np.array([sample_rate], dtype=np.int64)
        
        self.load_model()

    def load_model(self):
        try:
            import onnxruntime
            
            # Auto-download model if not present
            model_path = "models/silero_vad.onnx"
            if not os.path.exists(model_path):
                logging.info("⬇️ Downloading Silero VAD ONNX model...")
                os.makedirs("models", exist_ok=True)
                import urllib.request
                url = "https://github.com/snakers4/silero-vad/raw/master/files/silero_vad.onnx"
                urllib.request.urlretrieve(url, model_path)
                logging.info("✅ Silero VAD Downloaded")

            options = onnxruntime.SessionOptions()
            options.log_severity_level = 3
            # Use CPU by default for VAD (it's very light), unless CUDA requested
            self.session = onnxruntime.InferenceSession(model_path, options, providers=["CPUExecutionProvider"])
            logging.info(f"✅ Silero VAD Loaded (Sample Rate: {self.sample_rate})")
            
        except ImportError:
            logging.error("❌ 'onnxruntime' or 'numpy' not found. Please run: pip install onnxruntime numpy")
        except Exception as e:
            logging.error(f"❌ Failed to load Silero VAD: {e}")

    def reset_states(self):
        self._h = np.zeros((2, 1, 64), dtype=np.float32)
        self._c = np.zeros((2, 1, 64), dtype=np.float32)
        self.in_speech = False
        self.speech_buffer = bytearray()

    def process_chunk(self, chunk):
        self.buffer.extend(chunk)
        
        # We need exactly window_size_samples * 2 bytes (16-bit)
        required_bytes = self.window_size_samples * 2
        
        detected_utterance = None
        barge_in_triggered = False

        while len(self.buffer) >= required_bytes:
            # Extract frame
            frame_bytes = self.buffer[:required_bytes]
            self.buffer = self.buffer[required_bytes:]
            
            # Convert to float32 for ONNX
            input_tensor = np.frombuffer(frame_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            input_tensor = input_tensor.reshape(1, -1) # (1, window_size)

            speech_prob = 0.0
            
            if self.session:
                try:
                    ort_inputs = {
                        "input": input_tensor,
                        "sr": self._last_sr,
                        "h": self._h,
                        "c": self._c
                    }
                    ort_outs = self.session.run(None, ort_inputs)
                    speech_prob = ort_outs[0][0][0]
                    self._h, self._c = ort_outs[1], ort_outs[2]
                except Exception as e:
                    logging.error(f"VAD Inference Error: {e}")
            
            # Logic
            is_speech = speech_prob > self.threshold
            
            if is_speech:
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
                        # Optional: Reset RNN states here if desired
                        # self.reset_states() 
        
        if barge_in_triggered:
            return "BARGE_IN"
        
        return detected_utterance