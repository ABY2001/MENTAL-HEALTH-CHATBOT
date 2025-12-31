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

# Create tables
Base.metadata.create_all(bind=engine)

app = FastAPI(title="Mental Health Support API")

# Enable CORS for Angular
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4200"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load models
MODEL_PATH = "model/best_model.keras"
ENCODER_PATH = "model/label_encoder.pkl"

try:
    model = load_model(MODEL_PATH)
    with open(ENCODER_PATH, "rb") as f:
        label_encoder = pickle.load(f)
    print("✓ Emotion detection model loaded successfully")
except Exception as e:
    print(f"✗ Error loading model: {e}")
    model = None
    label_encoder = None

# ==================== PYDANTIC MODELS ====================
class LoginRequest(BaseModel):
    email: str
    password: str

class TextRequest(BaseModel):
    text: str

class EmotionResponse(BaseModel):
    emotion: str
    confidence: float
    all_emotions: dict

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
    """Predict emotion from audio file"""
    
    if model is None or label_encoder is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    temp_file = f"temp_{file.filename}"
    
    try:
        # Save uploaded audio
        content = await file.read()
        with open(temp_file, "wb") as f:
            f.write(content)
        
        print(f"Processing audio file: {temp_file}, size: {len(content)} bytes")
        
        # Feature extraction
        try:
            features = extract_mel_spectrogram(temp_file)
        except Exception as e:
            print(f"Feature extraction error: {e}")
            raise HTTPException(status_code=400, detail=f"Audio processing failed: {str(e)}")
        
        if features is None:
            raise HTTPException(status_code=400, detail="Failed to extract audio features")
        
        print(f"Features extracted, shape: {features.shape}")
        
        # Prepare input - check if model expects channel dimension
        features = np.expand_dims(features, axis=0)
        
        # Only add channel dimension if model expects it (4D input)
        if len(model.input_shape) == 4:
            features = np.expand_dims(features, axis=-1)
        
        print(f"Input shape for model: {features.shape}")
        
        # Predict
        predictions = model.predict(features, verbose=0)[0]
        idx = np.argmax(predictions)
        
        emotion = label_encoder.inverse_transform([idx])[0]
        confidence = float(predictions[idx])
        
        print(f"Prediction: {emotion} with confidence {confidence}")
        
        # Get all emotion probabilities
        all_emotions = {
            label_encoder.inverse_transform([i])[0]: float(predictions[i])
            for i in range(len(predictions))
        }
        
        return {
            "emotion": emotion,
            "confidence": round(confidence, 3),
            "all_emotions": all_emotions
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

@app.post("/predict-emotion-text", response_model=EmotionResponse)
async def predict_emotion_text(request: TextRequest):
    """
    Predict emotion from text using keyword analysis
    TODO: Replace with proper NLP model for better accuracy
    """
    text = request.text.lower().strip()
    
    if not text:
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    
    # Enhanced keyword-based emotion detection
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
    
    # Calculate scores for each emotion
    emotion_scores = {}
    for emotion, data in emotion_keywords.items():
        score = 0
        keywords = data['keywords']
        weight = data['weight']
        
        for keyword in keywords:
            if keyword in text:
                score += weight
        
        emotion_scores[emotion] = score
    
    # Determine dominant emotion
    detected_emotion = max(emotion_scores, key=emotion_scores.get)
    max_score = emotion_scores[detected_emotion]
    
    # If no keywords matched, default to neutral
    if max_score == 0:
        detected_emotion = 'neutral'
        confidence = 0.5
    else:
        # Calculate confidence (normalized between 0.6 and 0.95)
        confidence = min(0.6 + (max_score * 0.08), 0.95)
    
    # Normalize scores for all_emotions
    total_score = sum(emotion_scores.values()) or 1
    all_emotions = {
        emotion: round(score / total_score, 3) if total_score > 0 else 0.1
        for emotion, score in emotion_scores.items()
    }
    
    # Ensure detected emotion has highest probability
    all_emotions[detected_emotion] = max(all_emotions[detected_emotion], confidence)
    
    return {
        "emotion": detected_emotion,
        "confidence": round(confidence, 3),
        "all_emotions": all_emotions
    }

# ==================== CHAT HISTORY (OPTIONAL) ====================
class ChatMessage(BaseModel):
    user_id: int
    message: str
    emotion: str
    is_user: bool

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
    print("\n" + "="*70)
    print("🚀 Mental Health Support API Starting...")
    print("="*70)
    print(f"📍 API: http://127.0.0.1:8000")
    print(f"📚 Docs: http://127.0.0.1:8000/docs")
    print(f"🎭 Emotions: {list(label_encoder.classes_) if label_encoder else 'Model not loaded'}")
    print("="*70 + "\n")
    
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=True)


    