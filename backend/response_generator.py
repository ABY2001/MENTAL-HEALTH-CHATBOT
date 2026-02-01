"""
Response Generator
Generates empathetic, supportive responses based on emotion and safety assessment
WITH RAG-enhanced context and Ollama LLM integration
"""

from typing import Dict, List, Optional
from safety_engine import SafetyAssessment, RiskLevel, EmotionIntensity
import random

class ResponseGenerator:
    """
    Generates appropriate responses based on emotion, intensity, and safety level
    Enhanced with RAG system and Ollama LLM for natural responses
    """
    
    def __init__(self, rag_system=None, ollama_llm=None):
        """
        Initialize with optional RAG system and Ollama LLM
        
        Args:
            rag_system: RAGSystem instance for knowledge retrieval
            ollama_llm: OllamaLLM instance for response generation
        """
        self.rag_system = rag_system
        self.ollama_llm = ollama_llm
        # Response templates organized by emotion and intensity
        self.responses = {
            'happy': {
                'mild': [
                    "That's nice to hear! What's making you feel good today?",
                    "I'm glad you're having a positive moment. Tell me more about it.",
                    "It's wonderful that you're feeling good. What's been going well?"
                ],
                'moderate': [
                    "That's great! I'm really happy for you. What's bringing you joy?",
                    "Wonderful! It sounds like things are going well. What happened?",
                    "That's fantastic! Your positive energy is uplifting. Share more!"
                ],
                'severe': [
                    "That's amazing! I'm so glad you're feeling this happy. What's the source of your joy?",
                    "Incredible! Your happiness is contagious. Tell me all about what's making you feel this way!",
                    "That's absolutely wonderful! I'd love to hear more about what's bringing you such joy."
                ]
            },
            'sad': {
                'mild': [
                    "I hear that you're feeling a bit down. It's okay to feel this way. Would you like to talk about it?",
                    "I notice you're feeling sad. Remember, it's normal to have difficult emotions. What's on your mind?",
                    "I'm here for you. Feeling sad is a natural part of life. Want to share what's troubling you?"
                ],
                'moderate': [
                    "I'm sorry you're going through this difficult time. Your feelings are valid. Would you like to share more?",
                    "I can see this is really affecting you. Please know that you're not alone. What's been happening?",
                    "I hear your pain, and I want you to know I'm here to listen. Tell me more about what you're experiencing."
                ],
                'severe': [
                    "I'm deeply concerned about how you're feeling. Please know that you matter and your feelings are important. Can you tell me more about what's happening?",
                    "I hear that you're in a lot of pain right now. You don't have to go through this alone. I'm here to support you. What's been going on?",
                    "I want you to know that I'm here for you, and what you're feeling is valid. This pain won't last forever. Would you like to talk about what's troubling you?"
                ],
                'extreme': [
                    "I'm very concerned about you. Your wellbeing is important. Please know that there are people who care and want to help. Would it be okay if I share some resources with you?",
                    "I hear that you're in a really dark place right now. Please remember that you don't have to face this alone. There are trained professionals available 24/7 who want to help.",
                    "What you're going through sounds incredibly difficult. I want you to know that help is available, and things can get better. Would you be open to talking to a crisis counselor?"
                ]
            },
            'angry': {
                'mild': [
                    "I sense some frustration. It's completely okay to feel angry. What's bothering you?",
                    "I hear that something is irritating you. Your feelings are valid. Want to talk about it?",
                    "It sounds like something has upset you. Sometimes talking helps. What happened?"
                ],
                'moderate': [
                    "I can tell you're quite frustrated right now. Anger is a natural emotion. What's making you feel this way?",
                    "I understand you're feeling angry, and that's completely valid. Would you like to share what's causing this frustration?",
                    "Your anger is understandable. Sometimes we need to express these feelings. What's triggering this for you?"
                ],
                'severe': [
                    "I hear that you're really angry right now. These intense feelings can be overwhelming. Can you tell me what's causing this?",
                    "It sounds like you're experiencing a lot of anger. While these feelings are valid, let's work through them together. What happened?",
                    "I can sense your intense frustration. Let's take a moment together. What's making you feel this way?"
                ],
                'extreme': [
                    "I understand you're feeling extremely angry. Before we continue, I want to make sure you're safe. Are you in a place where you can take some deep breaths?",
                    "Your anger is very intense right now. Let's pause for a moment. Are you safe? Is there someone nearby who can support you?",
                    "I hear your rage. These feelings are overwhelming, but I want to help you process them safely. Can you tell me what's happening?"
                ]
            },
            'fearful': {
                'mild': [
                    "I notice you're feeling a bit worried. Anxiety is something we all experience. What's concerning you?",
                    "It sounds like you're feeling nervous about something. That's completely normal. Want to share what's on your mind?",
                    "I hear some worry in your words. Let's talk about what's making you feel this way."
                ],
                'moderate': [
                    "I can tell you're quite anxious right now. These feelings can be uncomfortable, but they will pass. What's causing this anxiety?",
                    "I understand you're feeling worried. Remember, you've gotten through difficult moments before. What's troubling you?",
                    "Your anxiety is valid. Let's work through this together. Can you tell me what's making you feel this way?"
                ],
                'severe': [
                    "I hear that you're experiencing intense anxiety. That must be really difficult. Remember to breathe. What's happening right now?",
                    "It sounds like you're feeling very scared or anxious. You're not alone in this. Can you describe what you're experiencing?",
                    "I'm concerned about your anxiety level. Let's focus on grounding. Can you name 5 things you can see around you? Then tell me what's frightening you."
                ],
                'extreme': [
                    "Your fear sounds overwhelming right now. I want to help you feel safe. Are you in immediate danger? If so, please call 911.",
                    "I hear that you're in extreme distress. Please know that you're not alone. Can you get to a safe space? Is there someone you trust nearby?",
                    "Your anxiety is at a critical level. I'm very concerned. Have you experienced panic attacks before? Would talking to a crisis counselor help?"
                ]
            },
            'calm': {
                'mild': [
                    "It's nice that you're feeling calm. What would you like to talk about?",
                    "I'm glad you're in a peaceful state. Is there anything on your mind?",
                    "That's good to hear. How can I support you today?"
                ],
                'moderate': [
                    "It's wonderful that you're feeling so calm and centered. What's contributing to this peace?",
                    "I'm glad you're experiencing this tranquility. What's been helping you feel this way?",
                    "That's great to hear. This calmness is a valuable state. What brought you here today?"
                ]
            },
            'neutral': {
                'mild': [
                    "I'm here to listen. What's on your mind today?",
                    "How can I support you right now?",
                    "Tell me what's happening in your life.",
                    "I'm here for whatever you need to talk about."
                ]
            },
            'surprised': {
                'mild': [
                    "That sounds unexpected! Tell me more about what happened.",
                    "It sounds like something caught you off guard. What was it?",
                    "Surprises can be intense. How are you processing this?"
                ]
            },
            'disgust': {
                'mild': [
                    "I hear that something is bothering you. What's making you feel uncomfortable?",
                    "It sounds like something doesn't sit right with you. Want to talk about it?",
                    "I sense you're disturbed by something. What's going on?"
                ]
            }
        }
        
        # Crisis responses
        self.crisis_responses = {
            RiskLevel.CRITICAL: [
                "I'm extremely worried about you right now. Your life has value, and there are people who want to help. Please call the National Suicide Prevention Lifeline at 988 immediately, or text HOME to 741741.",
                "What you're feeling right now is temporary, even though it doesn't feel that way. Please reach out to a crisis counselor right now at 988. They are trained to help and available 24/7.",
                "I care about your safety. Please don't face this alone. Call 988 or go to your nearest emergency room. You deserve help and support."
            ],
            RiskLevel.HIGH: [
                "I'm very concerned about what you're going through. Please consider reaching out to a mental health professional or calling the SAMHSA helpline at 1-800-662-4357.",
                "Your feelings are serious, and you deserve professional support. Would you consider calling a crisis helpline at 988? They can provide immediate help.",
                "What you're experiencing requires more support than I can provide. Please reach out to a mental health professional or crisis line at 988."
            ]
        }
    
    def generate_response(
        self, 
        emotion: str, 
        safety_assessment: SafetyAssessment,
        user_text: str = ""
    ) -> str:
        """
        Generate an appropriate response based on emotion and safety assessment
        Uses Ollama LLM if available, falls back to templates
        """
        # Handle crisis situations first (override LLM for safety)
        if safety_assessment.risk_level == RiskLevel.CRITICAL:
            crisis_msg = random.choice(self.crisis_responses[RiskLevel.CRITICAL])
            if safety_assessment.crisis_resources:
                resources = "\n\n📞 Resources:\n"
                for resource in safety_assessment.crisis_resources[:2]:
                    resources += f"• {resource['name']}: {resource['number']}\n"
                return crisis_msg + resources
            return crisis_msg
        
        if safety_assessment.risk_level == RiskLevel.HIGH:
            crisis_msg = random.choice(self.crisis_responses[RiskLevel.HIGH])
            response = crisis_msg
            
            if safety_assessment.crisis_resources:
                resources = "\n\n📞 Resources:\n"
                for resource in safety_assessment.crisis_resources[:2]:
                    resources += f"• {resource['name']}: {resource['number']}\n"
                response += resources
            
            return response
        
        # Try Ollama LLM first (for natural responses)
        if self.ollama_llm and self.ollama_llm.available:
            try:
                # Get RAG context if available
                context = ""
                if self.rag_system and user_text:
                    context = self.rag_system.get_context_for_emotion(emotion, user_text)
                
                # Generate response with LLM
                response = self.ollama_llm.generate_response(
                    user_message=user_text,
                    emotion=emotion,
                    intensity=safety_assessment.intensity.value,
                    risk_level=safety_assessment.risk_level.value,
                    context=context
                )
                
                # Add resources for medium risk
                if safety_assessment.risk_level == RiskLevel.MEDIUM and safety_assessment.crisis_resources:
                    response += "\n\nIf you need support: " + safety_assessment.crisis_resources[0]['number']
                
                return response
                
            except Exception as e:
                print(f"LLM generation failed: {e}, using fallback")
                # Fall through to template responses
        
        # Fallback to template responses
        intensity_key = safety_assessment.intensity.value
        
        if emotion in self.responses:
            if intensity_key in self.responses[emotion]:
                response = random.choice(self.responses[emotion][intensity_key])
            else:
                response = random.choice(self.responses[emotion].get('mild', ["I'm here to listen."]))
        else:
            response = "I'm here to support you. Tell me more about what you're experiencing."
        
        # Enhance with RAG context if available
        if self.rag_system and user_text:
            try:
                context = self.rag_system.get_context_for_emotion(emotion, user_text)
                if context:
                    response += f"\n\n💡 Here's something that might help:\n{context[:300]}..."
            except Exception as e:
                print(f"RAG retrieval error: {e}")
        
        # Add resources for medium risk
        if safety_assessment.risk_level == RiskLevel.MEDIUM and safety_assessment.crisis_resources:
            response += "\n\nIf you need additional support, these resources are available:\n"
            for resource in safety_assessment.crisis_resources[:1]:
                response += f"• {resource['name']}: {resource['number']}\n"
        
        return response

# Example usage
if __name__ == "__main__":
    from safety_engine import SafetyTriageEngine
    
    generator = ResponseGenerator()
    engine = SafetyTriageEngine()
    
    test_cases = [
        ("I feel a bit sad today", "sad", 0.65),
        ("I'm extremely depressed", "sad", 0.92),
        ("I want to end my life", "sad", 0.88),
        ("I'm having a great day!", "happy", 0.85),
        ("I'm so angry I could explode", "angry", 0.90),
    ]
    
    print("="*70)
    print("RESPONSE GENERATOR - TEST")
    print("="*70)
    
    for text, emotion, confidence in test_cases:
        print(f"\nUser: '{text}'")
        print(f"Emotion: {emotion} ({confidence})")
        
        assessment = engine.evaluate(text, emotion, confidence)
        response = generator.generate_response(emotion, assessment, text)
        
        print(f"Bot: {response}")
        print("-"*70)