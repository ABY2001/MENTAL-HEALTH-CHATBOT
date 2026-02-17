from datetime import datetime
from fastapi import FastAPI, Depends, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from sqlalchemy import desc
from pydantic import BaseModel
import numpy as np
import os
import pickle
from tensorflow.keras.models import load_model
import logging
from dotenv import load_dotenv

load_dotenv()

from database import Base, engine, SessionLocal
from models import User, ChatMessage
from audio_utils import extract_mel_spectrogram
from safety_engine import SafetyTriageEngine
from groq_service import ai

try:
    from audio_transcription import AudioTranscriptionService
    TRANSCRIPTION_AVAILABLE = True
except ImportError:
    TRANSCRIPTION_AVAILABLE = False

Base.metadata.create_all(bind=engine)
app = FastAPI(title="Mental Health Support API")
logging.getLogger("uvicorn.access").disabled = True

# ==================== INITIALIZATION ====================
print("\n" + "="*70)
print("🚀 SYSTEM STARTUP")
print("="*70)

safety_engine = SafetyTriageEngine()
print("✓ Safety Engine initialized")

transcription_service = None
if TRANSCRIPTION_AVAILABLE:
    try:
        transcription_service = AudioTranscriptionService(model_size="base")
        print("✓ Transcription Service initialized")
    except Exception as e:
        print(f"⚠️ Transcription failed: {e}")

# CORS setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4200"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["Content-Type"],
)

# Load Audio CNN Model
MODEL_PATH = "models/best_model.keras"
ENCODER_PATH = "models/label_encoder.pkl"
model = None
label_encoder = None

try:
    model = load_model(MODEL_PATH)
    with open(ENCODER_PATH, "rb") as f:
        label_encoder = pickle.load(f)
    print(f"✓ CNN Audio Model loaded (Classes: {list(label_encoder.classes_)})")
except Exception as e:
    print(f"⚠️ CNN Model NOT loaded: {e}")

# Data Models
class LoginRequest(BaseModel):
    email: str
    password: str

class TextRequest(BaseModel):
    text: str

class SaveChatRequest(BaseModel):
    """Request to save a chat message"""
    user_id: int
    user_message: str
    bot_response: str
    emotion: str
    emotion_confidence: float = 0.0


class ChatHistoryResponse(BaseModel):
    id: int
    user_message: str
    bot_response: str
    emotion: str
    emotion_confidence: float
    created_at: datetime

    class Config:
        from_attributes = True


class ChatSessionResponse(BaseModel):
    user_id: int
    message_count: int
    last_message: str
    last_emotion: str
    last_updated: datetime
    messages: list[ChatHistoryResponse]

# Database
def get_db():
    db = SessionLocal()
    try: 
        yield db
    finally: 
        db.close()

# ==================== ENDPOINTS ====================

@app.get("/")
def health_check():
    return {
        "status": "online",
        "groq_available": ai.available if hasattr(ai, 'available') else ai.client is not None,
        "transcription_available": TRANSCRIPTION_AVAILABLE,
        "cnn_model_loaded": model is not None,
        "emotions": list(label_encoder.classes_) if label_encoder else []
    }

@app.post("/predict-emotion-text")
def predict_emotion_text(request: TextRequest):
    """Text-based emotion detection using Groq"""
    text = request.text.strip()
    if not text: 
        raise HTTPException(status_code=400, detail="Empty text")
    
    print(f"\n📝 Text: '{text}'")
    
    try:
        # 1. Groq detects emotion from text
        emotion, confidence, _ = ai.detect_emotion(text)
        print(f"   Groq: {emotion} ({confidence:.2f})")
        
        # 2. Safety Check
        safety = safety_engine.evaluate(text, emotion, confidence)
        
        # 3. Groq generates response
        bot_response = ai.chat(text, emotion, safety.crisis_detected)
        if not bot_response:
            bot_response = "I'm here to listen."
        
        print(f"   Response: {bot_response[:50]}...")

        return {
            "emotion": emotion,
            "confidence": round(confidence, 3),
            "bot_response": bot_response,
            "safety": {
                "risk_level": safety.risk_level.value,
                "crisis_detected": safety.crisis_detected,
                "intensity": safety.intensity.value,
                "warning": safety.warning_message
            }
        }
    
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

# ==================== AUDIO ENDPOINT - CNN MODEL ====================
@app.post("/predict-emotion")
def predict_emotion(file: UploadFile = File(...)):
    """Audio emotion detection using CNN + Librosa model"""
    
    if model is None:
        raise HTTPException(status_code=500, detail="CNN model not loaded")
    
    temp_file = f"temp_{file.filename}"
    
    try:
        # Save audio
        content = file.file.read()
        with open(temp_file, "wb") as f:
            f.write(content)
        
        print(f"\n🎤 Audio: {len(content)} bytes")
        
        # 1. Extract mel-spectrogram using librosa
        print("   Extracting mel-spectrogram...")
        features = extract_mel_spectrogram(temp_file)
        
        if features is None:
            raise HTTPException(status_code=400, detail="Failed to extract audio features")
        
        print(f"   Features: shape={features.shape}")
        
        # 2. Prepare input for CNN model
        features_input = np.expand_dims(features, axis=0)  # Add batch dimension
        if len(model.input_shape) == 4:
            features_input = np.expand_dims(features_input, axis=-1)  # Add channel dimension
        
        print(f"   Model input shape: {features_input.shape}")
        
        # 3. CNN Prediction (this is your trained model!)
        print("   CNN inference...")
        predictions = model.predict(features_input, verbose=0)[0]
        
        # Get the predicted emotion
        idx = np.argmax(predictions)
        emotion = label_encoder.inverse_transform([idx])[0]
        confidence = float(predictions[idx])
        
        print(f"   ✅ CNN Result: {emotion} ({confidence:.2f})")
        print(f"   Prediction scores: {dict(zip(label_encoder.classes_, predictions))}")
        
        # 4. Optional: Get transcription for context
        transcription_text = ""
        if transcription_service:
            try:
                print("   Transcribing...")
                transcription = transcription_service.transcribe(temp_file)
                transcription_text = transcription.get('text', '')
                print(f"   Transcription: '{transcription_text}'")
            except Exception as e:
                print(f"   ⚠️ Transcription skipped: {e}")
        
        # 5. Safety assessment using CNN emotion
        analysis_text = transcription_text if transcription_text else "voice message"
        safety = safety_engine.evaluate(analysis_text, emotion, confidence)
        print(f"   Safety: crisis={safety.crisis_detected}")
        
        # 6. Generate response using Groq (with CNN-detected emotion)
        bot_response = ai.chat(analysis_text, emotion, safety.crisis_detected)
        if not bot_response:
            bot_response = "I'm listening."
        
        print(f"   Response: {bot_response[:50]}...")
        
        return {
            "emotion": emotion,
            "confidence": round(confidence, 3),
            "transcription": transcription_text,
            "bot_response": bot_response,
            "audio_quality": {"clear": True},
            "safety": {
                "risk_level": safety.risk_level.value,
                "crisis_detected": safety.crisis_detected,
                "intensity": safety.intensity.value,
                "warning": safety.warning_message
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
        
    finally:
        if os.path.exists(temp_file):
            try:
                os.remove(temp_file)
            except:
                pass

# ==================== AUTH ENDPOINTS ====================
@app.post("/login")
def login(request: LoginRequest, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == request.email.lower()).first()
    if not user or request.password != user.password:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    return {"message": "Login successful", "user_id": user.id, "email": user.email}

@app.post("/register")
def register(request: LoginRequest, db: Session = Depends(get_db)):
    if db.query(User).filter(User.email == request.email.lower()).first():
        raise HTTPException(status_code=400, detail="Email already registered")
    new_user = User(email=request.email.lower(), password=request.password)
    db.add(new_user)
    db.commit()
    return {"message": "Registration successful", "user_id": new_user.id, "email": new_user.email}

@app.get("/emotions")
def get_emotions():
    if label_encoder is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    return {"emotions": list(label_encoder.classes_)}

# ==================== CHAT HISTORY ENDPOINTS ====================

@app.post("/save-chat")
def save_chat(request: SaveChatRequest, db: Session = Depends(get_db)):
    """
    Save a chat message with emotion to database
    
    Example request:
    {
        "user_id": 1,
        "user_message": "I'm feeling great!",
        "bot_response": "That's wonderful to hear...",
        "emotion": "happy",
        "emotion_confidence": 0.87
    }
    """
    try:
        # Verify user exists
        user = db.query(User).filter(User.id == request.user_id).first()
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        
        # Create and save chat message
        chat_message = ChatMessage(
            user_id=request.user_id,
            user_message=request.user_message,
            bot_response=request.bot_response,
            emotion=request.emotion,
            emotion_confidence=request.emotion_confidence
        )
        
        db.add(chat_message)
        db.commit()
        db.refresh(chat_message)
        
        print(f"✓ Chat saved: User {request.user_id} | Emotion: {request.emotion}")
        
        return {
            "status": "saved",
            "chat_id": chat_message.id,
            "timestamp": chat_message.created_at
        }
    
    except Exception as e:
        db.rollback()
        print(f"❌ Error saving chat: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/chat-history/{user_id}")
def get_chat_history(user_id: int, db: Session = Depends(get_db)):
    """
    Get all chat history for a user
    Returns: List of all chats with user_id, ordered by newest first
    
    Example response:
    {
        "user_id": 1,
        "message_count": 5,
        "last_emotion": "happy",
        "messages": [
            {
                "id": 5,
                "user_message": "Thanks for listening",
                "bot_response": "You're welcome...",
                "emotion": "happy",
                "emotion_confidence": 0.85,
                "created_at": "2024-03-15T10:30:00"
            },
            ...
        ]
    }
    """
    try:
        # Verify user exists
        user = db.query(User).filter(User.id == user_id).first()
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        
        # Get all chat messages for this user, ordered by newest first
        chat_messages = db.query(ChatMessage).filter(
            ChatMessage.user_id == user_id
        ).order_by(desc(ChatMessage.created_at)).all()
        
        if not chat_messages:
            return {
                "user_id": user_id,
                "message_count": 0,
                "last_emotion": None,
                "last_updated": None,
                "messages": []
            }
        
        # Convert to response format
        messages_response = [
            ChatHistoryResponse.model_validate(msg) for msg in chat_messages
        ]
        
        last_emotion = chat_messages[0].emotion if chat_messages else None
        last_updated = chat_messages[0].created_at if chat_messages else None
        
        print(f"✓ Retrieved chat history for User {user_id}: {len(chat_messages)} messages")
        
        return {
            "user_id": user_id,
            "message_count": len(chat_messages),
            "last_emotion": last_emotion,
            "last_updated": last_updated,
            "messages": messages_response
        }
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error retrieving chat history: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/delete-chat-session/{user_id}")
def delete_chat_session(user_id: int, db: Session = Depends(get_db)):
    """
    Delete ALL chat messages for a user (entire chat session)
    
    Example response:
    {
        "status": "deleted",
        "user_id": 1,
        "messages_deleted": 15
    }
    """
    try:
        # Verify user exists
        user = db.query(User).filter(User.id == user_id).first()
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        
        # Count messages before deletion
        message_count = db.query(ChatMessage).filter(
            ChatMessage.user_id == user_id
        ).count()
        
        # Delete all messages for this user
        db.query(ChatMessage).filter(
            ChatMessage.user_id == user_id
        ).delete()
        
        db.commit()
        
        print(f"✓ Deleted chat session for User {user_id}: {message_count} messages removed")
        
        return {
            "status": "deleted",
            "user_id": user_id,
            "messages_deleted": message_count
        }
    
    except Exception as e:
        db.rollback()
        print(f"❌ Error deleting chat session: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/delete-chat-message/{message_id}")
def delete_chat_message(message_id: int, db: Session = Depends(get_db)):
    """
    Delete a single chat message by ID
    
    Example response:
    {
        "status": "deleted",
        "message_id": 5
    }
    """
    try:
        # Find and delete the message
        chat_message = db.query(ChatMessage).filter(
            ChatMessage.id == message_id
        ).first()
        
        if not chat_message:
            raise HTTPException(status_code=404, detail="Chat message not found")
        
        user_id = chat_message.user_id
        db.delete(chat_message)
        db.commit()
        
        print(f"✓ Deleted message {message_id} for User {user_id}")
        
        return {
            "status": "deleted",
            "message_id": message_id
        }
    
    except Exception as e:
        db.rollback()
        print(f"❌ Error deleting message: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/chat-stats/{user_id}")
def get_chat_stats(user_id: int, db: Session = Depends(get_db)):
    """
    Get statistics about user's chat history
    Returns: emotion distribution, total messages, avg confidence
    
    Example response:
    {
        "user_id": 1,
        "total_messages": 15,
        "avg_confidence": 0.82,
        "emotion_distribution": {
            "happy": 5,
            "calm": 4,
            "sad": 3,
            "anxious": 2,
            "angry": 1
        }
    }
    """
    try:
        # Get all messages for user
        messages = db.query(ChatMessage).filter(
            ChatMessage.user_id == user_id
        ).all()
        
        if not messages:
            return {
                "user_id": user_id,
                "total_messages": 0,
                "avg_confidence": 0.0,
                "emotion_distribution": {}
            }
        
        # Calculate statistics
        emotion_counts = {}
        total_confidence = 0
        
        for msg in messages:
            # Count emotions
            emotion = msg.emotion or "unknown"
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
            total_confidence += msg.emotion_confidence
        
        avg_confidence = total_confidence / len(messages) if messages else 0
        
        print(f"✓ Generated chat stats for User {user_id}")
        
        return {
            "user_id": user_id,
            "total_messages": len(messages),
            "avg_confidence": round(avg_confidence, 3),
            "emotion_distribution": emotion_counts
        }
    
    except Exception as e:
        print(f"❌ Error getting chat stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ==================== RUN ====================
if __name__ == "__main__":
    import uvicorn
    print("\n" + "="*70)
    print("🚀 Mental Health Support API")
    print("📍 http://127.0.0.1:8000")
    print("✓ Text: Groq LLM Analysis")
    print("✓ Audio: CNN + Librosa Model")
    print("="*70 + "\n")
    
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=False, access_log=False, log_level="warning")