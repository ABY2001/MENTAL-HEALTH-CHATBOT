from sqlalchemy import Column, Integer, String, DateTime, Text, ForeignKey, Float
from sqlalchemy.orm import relationship
from database import Base
from datetime import datetime
import uuid

class User(Base):
    __tablename__ = "user"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String(255), unique=True, index=True)
    password = Column(String(255))
    
    # Relationship to chat messages
    chat_messages = relationship("ChatMessage", back_populates="user", cascade="all, delete-orphan")


class ChatMessage(Base):
    """Store chat messages with emotions for history - Grouped by session"""
    __tablename__ = "chat_message"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("user.id"), index=True)
    
    # ⭐ NEW: Session ID to group messages together
    session_id = Column(String(100), default=lambda: str(uuid.uuid4()), index=True)
    
    # ⭐ NEW: Track message order within session
    turn_number = Column(Integer, default=1)
    
    # Message content
    user_message = Column(Text)  # What user typed/said
    bot_response = Column(Text)  # What bot responded
    
    # Emotion data
    emotion = Column(String(50))  # Detected emotion
    emotion_confidence = Column(Float, default=0.0)  # Confidence score
    
    # Timestamp
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    
    # Relationship back to user
    user = relationship("User", back_populates="chat_messages")
    
    def __repr__(self):
        return f"<ChatMessage(id={self.id}, user_id={self.user_id}, session_id={self.session_id}, turn={self.turn_number}, emotion={self.emotion})>"