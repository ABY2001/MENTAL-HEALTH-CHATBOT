from fastapi import FastAPI, Depends, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from pydantic import BaseModel
import numpy as np
import os
import pickle
from tensorflow.keras.models import load_model

from database import Base, engine, SessionLocal
from models import User
from auth import verify_password
from audio_utils import extract_mel_spectrogram
from safety_engine import SafetyTriageEngine, RiskLevel
from response_generator import ResponseGenerator

# Try to import RAG system (optional)
try:
    from rag_system import RAGSystem
    RAG_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  RAG system not available: {e}")
    print("Continuing without RAG enhancement...")
    RAG_AVAILABLE = False
    RAGSystem = None

# Try to import transcription service (optional)
try:
    from audio_transcription import AudioTranscriptionService
    TRANSCRIPTION_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  Transcription service not available: {e}")
    print("Install with: pip install openai-whisper")
    TRANSCRIPTION_AVAILABLE = False
    AudioTranscriptionService = None

# Try to import enhanced text emotion detector
try:
    from enhanced_text_emotion import EnhancedTextEmotionDetector
    TEXT_DETECTOR_AVAILABLE = True
except ImportError:
    print("⚠️ Enhanced text detector not available, using basic keyword matching")
    TEXT_DETECTOR_AVAILABLE = False
    EnhancedTextEmotionDetector = None

# Create tables
Base.metadata.create_all(bind=engine)

app = FastAPI(title="Mental Health Support API")

# Initialize Enhanced Text Emotion Detector
text_emotion_detector = None
if TEXT_DETECTOR_AVAILABLE:
    try:
        text_emotion_detector = EnhancedTextEmotionDetector("intents.json")
        print("✓ Enhanced Text Emotion Detector initialized")
    except Exception as e:
        print(f"⚠️ Could not initialize enhanced detector: {e}")

# Initialize RAG System (optional)
rag_system = None
if RAG_AVAILABLE:
    print("Initializing RAG System...")
    try:
        rag_system = RAGSystem()
    except Exception as e:
        print(f"⚠️  RAG system initialization failed: {e}")
        print("Continuing without RAG enhancement...")
        rag_system = None

# Initialize Transcription Service (optional)
transcription_service = None
if TRANSCRIPTION_AVAILABLE:
    print("Initializing Transcription Service...")
    try:
        transcription_service = AudioTranscriptionService(model_size="base")
    except Exception as e:
        print(f"⚠️  Transcription service initialization failed: {e}")
        print("Continuing without transcription...")
        transcription_service = None

# Initialize Safety Engine and Response Generator
safety_engine = SafetyTriageEngine()
response_generator = ResponseGenerator(rag_system=rag_system)
print("✓ Safety & Triage Engine initialized")
print("✓ Response Generator initialized")

# Enable CORS for Angular
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4200"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load models - Updated to correct path
MODEL_PATH = "models/best_model.keras"
ENCODER_PATH = "models/label_encoder.pkl"

try:
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found at: {os.path.abspath(MODEL_PATH)}")
    if not os.path.exists(ENCODER_PATH):
        raise FileNotFoundError(f"Encoder not found at: {os.path.abspath(ENCODER_PATH)}")
    
    model = load_model(MODEL_PATH)
    with open(ENCODER_PATH, "rb") as f:
        label_encoder = pickle.load(f)
    print("✓ Emotion detection model loaded successfully")
    print(f"  Model input shape: {model.input_shape}")
    print(f"  Available emotions: {list(label_encoder.classes_)}")
except Exception as e:
    print(f"✗ Error loading model: {e}")
    import traceback
    traceback.print_exc()
    model = None
    label_encoder = None

# ==================== PYDANTIC MODELS ====================
class LoginRequest(BaseModel):
    email: str
    password: str

class TextRequest(BaseModel):
    text: str

class ChatResponse(BaseModel):
    emotion: str
    confidence: float
    bot_response: str
    safety: dict

class ChatMessage(BaseModel):
    user_id: int
    message: str
    emotion: str
    is_user: bool

# ==================== DATABASE DEPENDENCY ====================
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# ==================== HEALTH CHECK ====================
@app.get("/")
def health_check():
    """API health check"""
    return {
        "status": "online",
        "model_loaded": model is not None,
        "available_emotions": list(label_encoder.classes_) if label_encoder else []
    }

# ==================== AUTHENTICATION ====================
@app.post("/login")
def login(request: LoginRequest, db: Session = Depends(get_db)):
    """User login endpoint"""
    clean_email = request.email.strip().lower()
    clean_password = request.password.strip()
    
    user = db.query(User).filter(User.email == clean_email).first()
    
    if not user:
        raise HTTPException(status_code=401, detail="Invalid email or password")
    
    # Use proper password verification if you have hashing
    # if not verify_password(clean_password, user.password):
    if clean_password != user.password:
        raise HTTPException(status_code=401, detail="Invalid email or password")
    
    return {
        "message": "Login successful",
        "user_id": user.id,
        "email": user.email,
        "name": getattr(user, 'name', None)
    }

@app.post("/register")
def register(request: LoginRequest, db: Session = Depends(get_db)):
    """User registration endpoint"""
    clean_email = request.email.strip().lower()
    clean_password = request.password.strip()
    
    # Check if user exists
    existing_user = db.query(User).filter(User.email == clean_email).first()
    if existing_user:
        raise HTTPException(status_code=400, detail="Email already registered")
    
    # Create new user (hash password in production)
    new_user = User(email=clean_email, password=clean_password)
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    
    return {
        "message": "Registration successful",
        "user_id": new_user.id,
        "email": new_user.email
    }

# ==================== EMOTION DETECTION ====================
@app.post("/predict-emotion")
async def predict_emotion(file: UploadFile = File(...)):
    """Predict emotion from audio file with transcription"""
    
    if model is None or label_encoder is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    temp_file = f"temp_{file.filename}"
    
    try:
        # Save uploaded audio
        content = await file.read()
        with open(temp_file, "wb") as f:
            f.write(content)
        
        print(f"Processing audio file: {temp_file}, size: {len(content)} bytes")
        
        # Step 1: Transcribe audio (if available)
        transcription = None
        transcription_text = ""
        
        if transcription_service:
            print("Transcribing audio...")
            transcription = transcription_service.transcribe(temp_file)
            transcription_text = transcription.get('text', '')
            
            print(f"Transcription: '{transcription_text}'")
            print(f"Confidence: {transcription.get('confidence', 0)}")
            
            # Check if audio is clear enough
            if not transcription_service.is_audio_clear(transcription, min_confidence=0.3):
                print("⚠️ Audio quality insufficient")
                return {
                    "emotion": "neutral",
                    "confidence": 0.0,
                    "transcription": transcription_text,
                    "bot_response": "I'm sorry, I couldn't hear you clearly. Could you please try recording again? Make sure you're in a quiet environment and speak clearly.",
                    "audio_quality": {
                        "clear": False,
                        "confidence": transcription.get('confidence', 0),
                        "issue": "Audio unclear or too quiet"
                    },
                    "safety": {
                        "risk_level": "low",
                        "crisis_detected": False,
                        "intensity": "mild",
                        "warning": ""
                    }
                }
        
        # Step 2: Feature extraction
        try:
            features = extract_mel_spectrogram(temp_file)
        except Exception as e:
            print(f"Feature extraction error: {e}")
            
            # If we have transcription, use it for emotion detection
            if transcription_text:
                print("Using transcription for emotion detection")
                emotion, confidence, all_emotions = detect_emotion_from_text(transcription_text)
                safety_assessment = safety_engine.evaluate(transcription_text, emotion, confidence)
                bot_response = response_generator.generate_response(emotion, safety_assessment, transcription_text)
                
                return {
                    "emotion": emotion,
                    "confidence": round(confidence, 3),
                    "transcription": transcription_text,
                    "bot_response": bot_response,
                    "audio_quality": {
                        "clear": True,
                        "method": "text_based",
                        "note": "Used transcription for emotion detection"
                    },
                    "safety": {
                        "risk_level": safety_assessment.risk_level.value,
                        "crisis_detected": safety_assessment.crisis_detected,
                        "intensity": safety_assessment.intensity.value,
                        "warning": safety_assessment.warning_message
                    }
                }
            else:
                raise HTTPException(status_code=400, detail=f"Audio processing failed: {str(e)}")
        
        if features is None:
            # Try text-based emotion detection if we have transcription
            if transcription_text:
                print("Features extraction failed, using transcription")
                emotion, confidence, all_emotions = detect_emotion_from_text(transcription_text)
                safety_assessment = safety_engine.evaluate(transcription_text, emotion, confidence)
                bot_response = response_generator.generate_response(emotion, safety_assessment, transcription_text)
                
                return {
                    "emotion": emotion,
                    "confidence": round(confidence, 3),
                    "transcription": transcription_text,
                    "bot_response": bot_response,
                    "audio_quality": {
                        "clear": True,
                        "method": "text_based",
                        "note": "Audio feature extraction failed, used text analysis"
                    },
                    "safety": {
                        "risk_level": safety_assessment.risk_level.value,
                        "crisis_detected": safety_assessment.crisis_detected,
                        "intensity": safety_assessment.intensity.value,
                        "warning": safety_assessment.warning_message
                    }
                }
            else:
                raise HTTPException(status_code=400, detail="Failed to extract audio features and no transcription available")
        
        print(f"Features extracted, shape: {features.shape}")
        
        # Step 3: Prepare input
        features_input = np.expand_dims(features, axis=0)
        
        if len(model.input_shape) == 4:
            features_input = np.expand_dims(features_input, axis=-1)
        
        print(f"Input shape for model: {features_input.shape}")
        
        # Step 4: Predict emotion from audio
        predictions = model.predict(features_input, verbose=0)[0]
        idx = np.argmax(predictions)
        
        emotion = label_encoder.inverse_transform([idx])[0]
        confidence = float(predictions[idx])
        
        print(f"Audio-based prediction: {emotion} with confidence {confidence}")
        
        # Step 5: Combine with text-based emotion if we have transcription
        analysis_text = transcription_text if transcription_text else "voice message"
        
        if transcription_text and len(transcription_text) > 5:
            # Get text-based emotion too
            text_emotion, text_confidence, _ = detect_emotion_from_text(transcription_text)
            print(f"Text-based prediction: {text_emotion} with confidence {text_confidence}")
            
            # Use higher confidence prediction
            if text_confidence > confidence:
                print("Using text-based emotion (higher confidence)")
                emotion = text_emotion
                confidence = text_confidence
        
        # Step 6: Safety evaluation
        safety_assessment = safety_engine.evaluate(analysis_text, emotion, confidence)
        
        # Step 7: Generate response
        bot_response = response_generator.generate_response(emotion, safety_assessment, analysis_text)
        
        return {
            "emotion": emotion,
            "confidence": round(confidence, 3),
            "transcription": transcription_text,
            "bot_response": bot_response,
            "audio_quality": {
                "clear": True,
                "confidence": transcription.get('confidence', 1.0) if transcription else 1.0
            },
            "safety": {
                "risk_level": safety_assessment.risk_level.value,
                "crisis_detected": safety_assessment.crisis_detected,
                "intensity": safety_assessment.intensity.value,
                "warning": safety_assessment.warning_message
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        error_detail = traceback.format_exc()
        print(f"Error details:\n{error_detail}")
        raise HTTPException(status_code=500, detail=f"Error processing audio: {str(e)}")
    finally:
        # Cleanup temp file
        if os.path.exists(temp_file):
            try:
                os.remove(temp_file)
            except:
                pass

@app.post("/predict-emotion-text")
async def predict_emotion_text(request: TextRequest):
    """
    Predict emotion from text with safety evaluation and generate response
    """
    text = request.text.strip()
    
    if not text:
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    
    # Step 1: Detect emotion
    emotion, confidence, all_emotions = detect_emotion_from_text(text)
    
    # Step 2: Safety evaluation
    safety_assessment = safety_engine.evaluate(text, emotion, confidence)
    
    # Step 3: Generate appropriate response
    bot_response = response_generator.generate_response(emotion, safety_assessment, text)
    
    # Step 4: Prepare full response
    response_data = {
        "emotion": emotion,
        "confidence": round(confidence, 3),
        "all_emotions": all_emotions,
        "bot_response": bot_response,
        "safety": {
            "risk_level": safety_assessment.risk_level.value,
            "crisis_detected": safety_assessment.crisis_detected,
            "intensity": safety_assessment.intensity.value,
            "warning": safety_assessment.warning_message
        }
    }
    
    return response_data

def detect_emotion_from_text(text: str):
    """
    Helper function to detect emotion from text
    Uses enhanced detector if available, falls back to basic
    """
    # Try enhanced detector first
    if text_emotion_detector:
        emotion, confidence, all_emotions = text_emotion_detector.detect_emotion(text)
        return emotion, confidence, all_emotions
    
    # Fallback to basic keyword matching
    text_lower = text.lower()
    
    emotion_keywords = {
        'happy': {
            'keywords': ['happy', 'joy', 'joyful', 'great', 'wonderful', 'excited', 'good', 
                        'excellent', 'amazing', 'fantastic', 'delighted', 'cheerful', 'pleased',
                        'glad', 'love', 'awesome', 'brilliant'],
            'weight': 1.0
        },
        'sad': {
            'keywords': ['sad', 'depressed', 'down', 'unhappy', 'miserable', 'terrible', 
                        'awful', 'cry', 'crying', 'upset', 'disappointed', 'hurt', 'lonely',
                        'hopeless', 'worthless', 'devastated'],
            'weight': 1.2
        },
        'angry': {
            'keywords': ['angry', 'mad', 'furious', 'annoyed', 'irritated', 'frustrated',
                        'rage', 'hate', 'pissed', 'outraged', 'bitter', 'resentful'],
            'weight': 1.1
        },
        'fearful': {
            'keywords': ['scared', 'afraid', 'anxious', 'worried', 'nervous', 'fear', 
                        'panic', 'terrified', 'frightened', 'uneasy', 'stressed', 'tense',
                        'threatened', 'insecure'],
            'weight': 1.2
        },
        'calm': {
            'keywords': ['calm', 'peaceful', 'relaxed', 'serene', 'tranquil', 'composed',
                        'content', 'comfortable', 'zen', 'mellow'],
            'weight': 0.9
        },
        'neutral': {
            'keywords': ['okay', 'fine', 'alright', 'normal', 'usual'],
            'weight': 0.8
        },
        'surprised': {
            'keywords': ['surprised', 'shocked', 'amazed', 'astonished', 'unexpected',
                        'wow', 'incredible', 'unbelievable'],
            'weight': 0.9
        },
        'disgust': {
            'keywords': ['disgusted', 'revolted', 'gross', 'nasty', 'sick', 'repulsive'],
            'weight': 1.0
        }
    }
    
    emotion_scores = {}
    for emotion_name, data in emotion_keywords.items():
        score = 0
        keywords = data['keywords']
        weight = data['weight']
        
        for keyword in keywords:
            if keyword in text_lower:
                score += weight
        
        emotion_scores[emotion_name] = score
    
    detected_emotion = max(emotion_scores, key=emotion_scores.get)
    max_score = emotion_scores[detected_emotion]
    
    if max_score == 0:
        detected_emotion = 'neutral'
        confidence = 0.5
    else:
        confidence = min(0.6 + (max_score * 0.08), 0.95)
    
    total_score = sum(emotion_scores.values()) or 1
    all_emotions = {
        emotion_name: round(score / total_score, 3) if total_score > 0 else 0.1
        for emotion_name, score in emotion_scores.items()
    }
    
    all_emotions[detected_emotion] = max(all_emotions[detected_emotion], confidence)
    
    return detected_emotion, confidence, all_emotions

# ==================== CHAT HISTORY (OPTIONAL) ====================
@app.post("/save-message")
def save_message(message: ChatMessage, db: Session = Depends(get_db)):
    """Save chat message to database (optional feature)"""
    # You can implement this if you want to store chat history
    # For now, just return success
    return {"message": "Message saved successfully"}

# ==================== CRISIS DETECTION ====================
@app.post("/check-crisis")
def check_crisis(request: TextRequest):
    """
    Check if text contains crisis indicators
    Returns risk level and recommended actions
    """
    text = request.text.lower()
    
    # Crisis keywords (suicide, self-harm, violence)
    high_risk_keywords = [
        'suicide', 'kill myself', 'end my life', 'want to die', 'better off dead',
        'self harm', 'hurt myself', 'cut myself', 'overdose'
    ]
    
    medium_risk_keywords = [
        'hopeless', 'worthless', 'no point', 'give up', 'can\'t go on',
        'unbearable', 'escape', 'end it all'
    ]
    
    # Check for high risk
    high_risk = any(keyword in text for keyword in high_risk_keywords)
    medium_risk = any(keyword in text for keyword in medium_risk_keywords)
    
    if high_risk:
        return {
            "risk_level": "high",
            "crisis_detected": True,
            "message": "Immediate support needed. Please reach out to a crisis helpline.",
            "resources": [
                {"name": "National Suicide Prevention Lifeline", "number": "988"},
                {"name": "Crisis Text Line", "number": "Text HOME to 741741"}
            ]
        }
    elif medium_risk:
        return {
            "risk_level": "medium",
            "crisis_detected": True,
            "message": "We're concerned about you. Consider talking to a mental health professional.",
            "resources": [
                {"name": "Mental Health Hotline", "number": "1-800-662-4357"}
            ]
        }
    else:
        return {
            "risk_level": "low",
            "crisis_detected": False,
            "message": "Continue conversation normally"
        }

# ==================== GET EMOTIONS ====================
@app.get("/emotions")
def get_emotions():
    """Get list of available emotions"""
    if label_encoder is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    return {
        "emotions": list(label_encoder.classes_)
    }

# ==================== RUN SERVER ====================
if __name__ == "__main__":
    import uvicorn
    import logging
    
    # Set logging to DEBUG to see all errors
    logging.basicConfig(level=logging.DEBUG)
    
    print("\n" + "="*70)
    print("🚀 Mental Health Support API Starting...")
    print("="*70)
    print(f"📍 API: http://127.0.0.1:8000")
    print(f"📚 Docs: http://127.0.0.1:8000/docs")
    print(f"🎭 Emotions: {list(label_encoder.classes_) if label_encoder else 'Model not loaded'}")
    print(f"📦 Model input shape: {model.input_shape if model else 'No model'}")
    print("="*70 + "\n")
    
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=False, log_level="debug")