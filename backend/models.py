from sqlalchemy import Column, Integer, String, DateTime, Text, ForeignKey, Float
from sqlalchemy.orm import relationship
from database import Base
from datetime import datetime

class User(Base):
    __tablename__ = "user"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String(255), unique=True, index=True)
    password = Column(String(255))
    
    # Relationship to chat messages
    chat_messages = relationship("ChatMessage", back_populates="user", cascade="all, delete-orphan")


class ChatMessage(Base):
    """Store chat messages with emotions for history"""
    __tablename__ = "chat_message"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("user.id"), index=True)
    
    # Message content
    user_message = Column(Text)  # What user typed/said
    bot_response = Column(Text)  # What bot responded
    
    # Emotion data
    emotion = Column(String(50))  # Detected emotion
    emotion_confidence = Column(Float, default=0.0)  # Confidence score (FIXED: Float with capital F)
    
    # Timestamp
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    
    # Relationship back to user
    user = relationship("User", back_populates="chat_messages")
    
    def __repr__(self):
        return f"<ChatMessage(id={self.id}, user_id={self.user_id}, emotion={self.emotion})>"