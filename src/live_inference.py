import os
import time
import cv2
import torch
import numpy as np
import wave
try:
    import sounddevice as sd
except ImportError:
    print("WARNING: 'sounddevice' not found. Audio will not be recorded.")
    print("Please run: pip install sounddevice scipy")
    sd = None

import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.streaming_drcfnet import StreamingDRCFNet

# ==========================================
# CONFIGURATION
# ==========================================
MODEL_WEIGHTS = "src/models/drcfnet_mosi_best.pt"
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
AUDIO_RATE = 16000
CHUNK_DURATION = 2.0  # seconds to capture per loop

def main():
    print("==================================================")
    print(" DRCFNet Live Streaming Inference (Governance + Shadow)")
    print("==================================================")
    
    print(f"[*] Loading model on {DEVICE}...")
    model = StreamingDRCFNet(dim_v=35, dim_a=74, dim_t=300).to(DEVICE)
    
    # Try to load weights if they exist
    if os.path.exists(MODEL_WEIGHTS):
        checkpoint = torch.load(MODEL_WEIGHTS, map_location=DEVICE)
        # Handle dict or raw state_dict
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        else:
            model.load_state_dict(checkpoint, strict=False)
        print(f"[*] Successfully loaded weights from {MODEL_WEIGHTS}")
    else:
        print(f"[!] Warning: Weights {MODEL_WEIGHTS} not found. Running with random weights.")
        
    model.eval()
    
    # Try importing ImageBind preprocessors
    try:
        from imagebind import data
        from imagebind.models.imagebind_model import ModalityType
        IMAGEBIND_AVAILABLE = True
        print("[*] Meta ImageBind detected. Shadow Proxy Generation is ACTIVE.")
    except ImportError:
        IMAGEBIND_AVAILABLE = False
        print("[!] Meta ImageBind NOT detected. Fallback proxies will be zeros.")
        print("    To test the full Governance pipeline, install ImageBind:")
        print("    pip install git+https://github.com/facebookresearch/ImageBind.git")

    print("[*] Initializing Webcam...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[!] ERROR: Could not open webcam.")
        return

    print(f"[*] Starting Live Inference Loop (Press 'q' in video window to exit)...")
    
    while True:
        # 1. Capture Video Frame
        ret, frame = cap.read()
        if not ret:
            print("[!] Failed to grab frame.")
            break
            
        cv2.imshow('DRCFNet Live Feed', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
        # 2. Capture Audio Chunk
        if sd is not None:
            # print(f"Recording {CHUNK_DURATION}s audio...")
            audio_data = sd.rec(int(CHUNK_DURATION * AUDIO_RATE), samplerate=AUDIO_RATE, channels=1, dtype='int16')
            sd.wait()
        else:
            audio_data = np.zeros((int(CHUNK_DURATION * AUDIO_RATE), 1), dtype='int16')

        # Save temporary files for ImageBind loader (it natively expects files)
        temp_img_path = "tmp_vision.jpg"
        temp_aud_path = "tmp_audio.wav"
        
        cv2.imwrite(temp_img_path, frame)
        with wave.open(temp_aud_path, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2) # 2 bytes for int16
            wf.setframerate(AUDIO_RATE)
            wf.writeframes(audio_data.tobytes())
        
        # 3. Construct the Raw Inputs for Thread B (ImageBind Proxy)
        raw_inputs = None
        if IMAGEBIND_AVAILABLE:
            # ImageBind's sophisticated data loaders handle the complex Mel-spectrograms and vision crops
            raw_inputs = {
                ModalityType.VISION: data.load_and_transform_vision_data([temp_img_path], DEVICE),
                ModalityType.AUDIO: data.load_and_transform_audio_data([temp_aud_path], DEVICE)
                # Text proxy is skipped for live unless we hook up a live ASR (e.g., Whisper)
            }
            
        # 4. Construct Dummy Inputs for Thread A (Simulating complete modality failure/noise)
        # Because we are testing the Governance Layer, we intentionally feed Zeros (bad data)
        # to the real MS-TCN thread. The Governance Layer will detect the high entropy
        # and seamlessly fall back to the ImageBind proxies we generated above!
        dummy_vision = torch.zeros(1, 1, 35).to(DEVICE)
        dummy_audio = torch.zeros(1, 1, 74).to(DEVICE)
        dummy_text = torch.zeros(1, 1, 300).to(DEVICE)
        
        # 5. Forward Pass
        with torch.no_grad():
            output = model(
                vision=dummy_vision,
                audio=dummy_audio,
                text=dummy_text,
                raw_inputs=raw_inputs,
                kg_features=None, # Uses learned fallback node
                enable_shadow_training=False # Inference mode
            )
            
        prediction = output.item()
        
        # Print Result
        sentiment = "Positive" if prediction > 0 else "Negative"
        intensity = abs(prediction)
        print(f"Live Prediction: {prediction:+.3f} | Sentiment: {sentiment} (Intensity: {intensity:.2f})")
        
        # Clean up temps
        if os.path.exists(temp_img_path): os.remove(temp_img_path)
        if os.path.exists(temp_aud_path): os.remove(temp_aud_path)

    cap.release()
    cv2.destroyAllWindows()
    print("[*] Live Inference Terminated.")

if __name__ == "__main__":
    main()
