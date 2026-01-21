"""
Audio Transcription Service using OpenAI Whisper
Converts speech to text for better context and user feedback
"""

import os
import torch
import whisper
import numpy as np
from typing import Optional, Dict

class AudioTranscriptionService:
    """
    Transcribes audio files to text using Whisper model
    """
    
    def __init__(self, model_size: str = "base"):
        """
        Initialize Whisper model
        
        Args:
            model_size: Size of Whisper model
                - tiny: Fastest, least accurate (~1GB RAM)
                - base: Good balance (~1GB RAM) - RECOMMENDED
                - small: Better accuracy (~2GB RAM)
                - medium: High accuracy (~5GB RAM)
                - large: Best accuracy (~10GB RAM)
        """
        print(f"Loading Whisper model ({model_size})...")
        
        try:
            # Check if CUDA is available
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"Using device: {self.device}")
            
            # Load Whisper model
            self.model = whisper.load_model(model_size, device=self.device)
            print(f"✓ Whisper model loaded successfully")
            
        except Exception as e:
            print(f"✗ Error loading Whisper model: {e}")
            self.model = None
    
    def transcribe(self, audio_path: str, language: str = "en") -> Dict:
        """
        Transcribe audio file to text
        
        Args:
            audio_path: Path to audio file
            language: Language code (e.g., 'en' for English)
        
        Returns:
            Dictionary with transcription results:
            {
                'text': 'transcribed text',
                'success': True/False,
                'confidence': 0.0-1.0,
                'language': 'detected language',
                'error': 'error message if failed'
            }
        """
        if self.model is None:
            return {
                'text': '',
                'success': False,
                'confidence': 0.0,
                'language': language,
                'error': 'Whisper model not loaded'
            }
        
        try:
            # Transcribe
            result = self.model.transcribe(
                audio_path,
                language=language,
                fp16=False,  # Use fp32 for CPU compatibility
                verbose=False
            )
            
            text = result.get('text', '').strip()
            detected_language = result.get('language', language)
            
            # Calculate rough confidence from segments if available
            segments = result.get('segments', [])
            if segments:
                # Average the no_speech_prob (lower is better)
                avg_no_speech = np.mean([
                    seg.get('no_speech_prob', 0.5) 
                    for seg in segments
                ])
                confidence = 1.0 - avg_no_speech
            else:
                confidence = 0.8 if text else 0.0
            
            return {
                'text': text,
                'success': True,
                'confidence': round(float(confidence), 3),
                'language': detected_language,
                'error': None
            }
            
        except Exception as e:
            print(f"Transcription error: {e}")
            return {
                'text': '',
                'success': False,
                'confidence': 0.0,
                'language': language,
                'error': str(e)
            }
    
    def is_audio_clear(self, transcription_result: Dict, min_confidence: float = 0.5) -> bool:
        """
        Check if audio is clear enough for processing
        
        Args:
            transcription_result: Result from transcribe()
            min_confidence: Minimum confidence threshold
        
        Returns:
            True if audio is clear, False otherwise
        """
        if not transcription_result['success']:
            return False
        
        # Check if we got any text
        if not transcription_result['text']:
            return False
        
        # Check confidence
        if transcription_result['confidence'] < min_confidence:
            return False
        
        # Check minimum length (at least 3 characters)
        if len(transcription_result['text']) < 3:
            return False
        
        return True

# Fallback for when Whisper is not available
class SimpleAudioQualityChecker:
    """
    Simple audio quality checker without transcription
    Checks basic audio properties
    """
    
    def __init__(self):
        print("✓ Simple audio quality checker initialized")
    
    def check_audio_quality(self, audio_path: str) -> Dict:
        """
        Check if audio file is valid and has content
        """
        import librosa
        
        try:
            # Load audio
            y, sr = librosa.load(audio_path, sr=16000)
            
            # Check duration
            duration = len(y) / sr
            if duration < 0.3:
                return {
                    'valid': False,
                    'error': 'Audio too short (minimum 0.3 seconds)',
                    'duration': duration
                }
            
            # Check if mostly silence
            rms = librosa.feature.rms(y=y)
            energy = np.mean(rms)
            
            if energy < 0.01:
                return {
                    'valid': False,
                    'error': 'Audio appears to be silent or very quiet',
                    'duration': duration
                }
            
            return {
                'valid': True,
                'error': None,
                'duration': duration,
                'energy': float(energy)
            }
            
        except Exception as e:
            return {
                'valid': False,
                'error': f'Audio processing failed: {str(e)}',
                'duration': 0
            }

# Example usage and testing
if __name__ == "__main__":
    import sys
    
    # Test with Whisper
    try:
        transcriber = AudioTranscriptionService(model_size="base")
        
        # Test file
        if len(sys.argv) > 1:
            test_file = sys.argv[1]
        else:
            test_file = "temp_recording.webm"
        
        if os.path.exists(test_file):
            print(f"\nTranscribing: {test_file}")
            result = transcriber.transcribe(test_file)
            
            print("\n" + "="*70)
            print("TRANSCRIPTION RESULT:")
            print("="*70)
            print(f"Success: {result['success']}")
            print(f"Text: '{result['text']}'")
            print(f"Confidence: {result['confidence']}")
            print(f"Language: {result['language']}")
            print(f"Clear audio: {transcriber.is_audio_clear(result)}")
            
            if result['error']:
                print(f"Error: {result['error']}")
        else:
            print(f"Test file not found: {test_file}")
            
    except Exception as e:
        print(f"Error: {e}")
        print("\nFalling back to simple quality checker...")
        
        checker = SimpleAudioQualityChecker()
        if os.path.exists(test_file):
            quality = checker.check_audio_quality(test_file)
            print(f"Audio quality check: {quality}")