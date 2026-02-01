"""
Ollama LLM Integration
Provides natural, empathetic responses using local AI models
"""

import requests
from typing import Dict, Optional
import json

class OllamaLLM:
    """
    Interface to Ollama for generating mental health support responses
    """
    
    def __init__(
        self, 
        base_url: str = "http://localhost:11434",
        model: str = "llama3.2"
    ):
        """
        Initialize Ollama client
        
        Args:
            base_url: Ollama API endpoint
            model: Model to use (llama3.2, mistral, etc.)
        """
        self.base_url = base_url
        self.model = model
        self.available = self._check_availability()
        
        if self.available:
            print(f"✓ Ollama connected: {model}")
        else:
            print(f"⚠️  Ollama not available at {base_url}")
    
    def _check_availability(self) -> bool:
        """Check if Ollama is running and model is available"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=30)
            if response.status_code == 200:
                models = response.json().get('models', [])
                model_names = [m.get('name', '') for m in models]
                
                # Check if our model is available
                if any(self.model in name for name in model_names):
                    return True
                else:
                    print(f"⚠️  Model '{self.model}' not found. Available: {model_names}")
                    return False
            return False
        except Exception as e:
            print(f"⚠️  Ollama connection failed: {e}")
            return False
    
    def generate_response(
        self,
        user_message: str,
        emotion: str,
        intensity: str,
        risk_level: str,
        context: str = "",
        conversation_history: list = None
    ) -> str:
        """
        Generate empathetic response using Ollama
        
        Args:
            user_message: User's message
            emotion: Detected emotion
            intensity: Emotion intensity (mild, moderate, severe, extreme)
            risk_level: Safety risk level (low, medium, high, critical)
            context: Additional context from RAG
            conversation_history: Previous messages for context
        
        Returns:
            Generated response
        """
        if not self.available:
            return self._fallback_response(emotion, intensity)
        
        # Build system prompt
        system_prompt = self._build_system_prompt(emotion, intensity, risk_level)
        
        # Build user prompt
        user_prompt = self._build_user_prompt(
            user_message, 
            emotion, 
            intensity,
            context,
            conversation_history
        )
        
        try:
            # Call Ollama API
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": user_prompt,
                    "system": system_prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.7,
                        "top_p": 0.9,
                        "max_tokens": 200
                    }
                },
                timeout=10
            )
            
            if response.status_code == 200:
                result = response.json()
                generated_text = result.get('response', '').strip()
                
                # Clean up response
                generated_text = self._clean_response(generated_text)
                
                return generated_text
            else:
                print(f"Ollama API error: {response.status_code}")
                return self._fallback_response(emotion, intensity)
                
        except requests.exceptions.Timeout:
            print("Ollama timeout - using fallback")
            return self._fallback_response(emotion, intensity)
        except Exception as e:
            print(f"Ollama error: {e}")
            return self._fallback_response(emotion, intensity)
    
    def _build_system_prompt(self, emotion: str, intensity: str, risk_level: str) -> str:
        """Build system prompt based on context"""
        
        base_prompt = """You are a compassionate mental health support assistant. Your role is to:
- Provide empathetic, supportive responses
- Validate the user's feelings
- Ask open-ended questions to encourage sharing
- Offer practical coping strategies when appropriate
- Be warm, non-judgmental, and understanding
- Keep responses concise (2-4 sentences)
- NEVER diagnose or replace professional therapy
- Use simple, clear language"""

        # Add emotion-specific guidance
        emotion_guidance = {
            'sad': "\nThe user is feeling sad. Acknowledge their pain, validate their feelings, and gently encourage them to share more.",
            'fearful': "\nThe user is anxious or scared. Help them feel safe, offer grounding techniques if severe, and show understanding.",
            'angry': "\nThe user is frustrated or angry. Validate their anger as normal, help them express it safely, and listen without judgment.",
            'happy': "\nThe user is feeling positive. Share in their joy, ask what's making them happy, and encourage them to savor it.",
            'neutral': "\nThe user seems calm. Be a good listener and let them guide the conversation."
        }
        
        # Add intensity guidance
        if intensity in ['severe', 'extreme']:
            base_prompt += "\n\nIMPORTANT: The user is experiencing INTENSE emotions. Be extra supportive and gentle."
        
        # Add risk guidance
        if risk_level == 'critical':
            base_prompt += "\n\nCRITICAL: This appears to be a crisis. Encourage immediate professional help while being supportive."
        elif risk_level == 'high':
            base_prompt += "\n\nHIGH RISK: Gently suggest professional support while being empathetic."
        
        return base_prompt + emotion_guidance.get(emotion, "")
    
    def _build_user_prompt(
        self, 
        user_message: str, 
        emotion: str,
        intensity: str,
        context: str,
        conversation_history: list
    ) -> str:
        """Build user prompt with context"""
        
        prompt = f"User (feeling {emotion}, {intensity} intensity): {user_message}\n\n"
        
        # Add RAG context if available
        if context:
            prompt += f"Relevant information:\n{context[:200]}...\n\n"
        
        prompt += "Respond with empathy and support (2-4 sentences):"
        
        return prompt
    
    def _clean_response(self, text: str) -> str:
        """Clean up generated response"""
        # Remove common artifacts
        text = text.replace("Here's my response:", "")
        text = text.replace("Response:", "")
        text = text.replace("Assistant:", "")
        
        # Remove extra whitespace
        text = " ".join(text.split())
        
        # Ensure ends with punctuation
        if text and text[-1] not in '.!?':
            text += "."
        
        return text.strip()
    
    def _fallback_response(self, emotion: str, intensity: str) -> str:
        """Fallback when Ollama unavailable"""
        responses = {
            'sad': "I hear that you're feeling down. It's completely okay to feel this way. Would you like to share what's troubling you?",
            'fearful': "I understand you're feeling anxious. These feelings can be overwhelming, but you're not alone. Can you tell me more about what's worrying you?",
            'angry': "I hear your frustration, and your feelings are valid. It's okay to be angry. What's causing these feelings?",
            'happy': "That's wonderful to hear! I'm glad you're feeling positive. What's bringing you joy today?",
            'neutral': "I'm here to listen. What's on your mind today?"
        }
        
        return responses.get(emotion, "I'm here for you. Tell me more about how you're feeling.")

# Example usage and testing
if __name__ == "__main__":
    # Test Ollama connection
    llm = OllamaLLM(model="llama3.2")
    
    if llm.available:
        test_cases = [
            {
                "message": "I got bad marks in my exam and feel terrible",
                "emotion": "sad",
                "intensity": "moderate",
                "risk": "low"
            },
            {
                "message": "I'm so anxious I can't breathe",
                "emotion": "fearful",
                "intensity": "severe",
                "risk": "medium"
            },
            {
                "message": "I'm feeling really happy today!",
                "emotion": "happy",
                "intensity": "moderate",
                "risk": "low"
            }
        ]
        
        print("\n" + "="*70)
        print("OLLAMA LLM - RESPONSE GENERATION TEST")
        print("="*70)
        
        for test in test_cases:
            print(f"\nUser: {test['message']}")
            print(f"Emotion: {test['emotion']} ({test['intensity']})")
            
            response = llm.generate_response(
                test['message'],
                test['emotion'],
                test['intensity'],
                test['risk']
            )
            
            print(f"Bot: {response}")
            print("-"*70)
    else:
        print("\n⚠️  Ollama not available. Install from: https://ollama.ai")
        print("Then run: ollama pull llama3.2")