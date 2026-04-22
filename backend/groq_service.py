import os
import json
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

# ==================== CRISIS SYSTEM PROMPT ====================
CRISIS_SYSTEM_PROMPT = """You are a crisis counselor responding to someone who may be suicidal or in severe distress.

YOU MUST DO ALL OF THE FOLLOWING IN YOUR RESPONSE:
1. Acknowledge their pain with genuine empathy — do NOT be dismissive or generic
2. Clearly say you are concerned for their safety
3. Include these exact crisis helpline numbers:
   - iCall: 9152987821 (24/7, free, Hindi/English/Marathi)
   - AASRA: 9820466726 (24/7, free)
   - Vandrevala Foundation: 9999776666 (24/7, free)
   - Emergency: 112
4. Urge them to call one of these numbers RIGHT NOW
5. Remind them they are not alone

DO NOT give generic comfort like "things will get better" without also providing the numbers.
DO NOT ignore the crisis. Keep response warm, direct, and under 130 words."""

# ==================== NORMAL SUPPORT PROMPT ====================
NORMAL_SYSTEM_PROMPT = """You are a warm, empathetic mental health support assistant.
- Validate the user's feelings without minimizing them
- Be non-judgmental and supportive
- Keep responses to 1-2 short sentences
- Do not give medical advice or diagnoses
- Focus on emotional support, not problem-solving unless asked"""


class GroqService:
    def __init__(self):
        self.api_key = os.getenv("GROQ_API_KEY")
        self.client = None
        self.model = "llama-3.1-8b-instant"
        
        # Simple greeting words (detect and classify as NEUTRAL)
        self.simple_greetings = ['hi', 'hello', 'hey', 'hiya', 'greetings', 'wassup', 'yo', 'sup', 'howdy', 'hii', 'hiii']
        
        if not self.api_key:
            print("❌ GROQ ERROR: No GROQ_API_KEY found in .env")
        else:
            try:
                self.client = Groq(api_key=self.api_key)
                self.client.models.list()
                print(f"✓ Groq Service Online (Model: {self.model})")
            except Exception as e:
                print(f"❌ Groq Connection Failed: {e}")
                self.client = None

    def _is_simple_greeting(self, text: str) -> bool:
        """Check if text is only a simple greeting (with or without punctuation)"""
        text_clean = ''.join(c for c in text.lower() if c.isalpha() or c.isspace()).strip()
        return text_clean in self.simple_greetings

    def detect_emotion(self, text: str):
        """
        Detect emotion from text
        Returns: (emotion, confidence, metadata)
        """
        if not self.client: 
            return "neutral", 0.0, {}

        # Check if input is a simple greeting first
        if self._is_simple_greeting(text):
            return "neutral", 0.95, {"is_greeting": True}

        prompt = f"""Analyze the emotion of this text: "{text}"
        Return ONLY valid JSON: {{ "emotion": "happy|sad|angry|neutral|fearful|calm|disgusted|surprised", "confidence": 0.9 }}
        Do not explain. Only return JSON."""

        try:
            chat_completion = self.client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=self.model,
                response_format={"type": "json_object"},
                temperature=0.2,
            )
            
            content = chat_completion.choices[0].message.content
            res = json.loads(content)
            emotion = res.get('emotion', 'neutral').lower()
            confidence = float(res.get('confidence', 0.5))
            
            return emotion, confidence, {}
        
        except Exception as e:
            print(f"⚠️ Emotion Error: {e}")
            return "neutral", 0.0, {}

    def chat(self, user_text: str, emotion: str, is_crisis: bool):
        """
        Generate response based on user emotion and message.
        If is_crisis=True, uses CRISIS_SYSTEM_PROMPT which forces hotline numbers into response.
        """
        if not self.client:
            return self._fallback_response(is_crisis)

        # Special response for simple greetings
        if self._is_simple_greeting(user_text):
            return "Hello! I'm here to support you. How are you feeling today?"

        try:
            if is_crisis:
                # CRISIS PATH: system prompt forces LLM to include hotline numbers
                chat_completion = self.client.chat.completions.create(
                    messages=[
                        {"role": "system", "content": CRISIS_SYSTEM_PROMPT},
                        {"role": "user", "content": (
                            f"The user said: \"{user_text}\"\n"
                            f"Detected emotion: {emotion}\n"
                            f"Please respond with empathy and provide crisis resources immediately."
                        )}
                    ],
                    model=self.model,
                    temperature=0.6,
                    max_tokens=250
                )
            else:
                # NORMAL PATH: warm supportive response
                chat_completion = self.client.chat.completions.create(
                    messages=[
                        {"role": "system", "content": NORMAL_SYSTEM_PROMPT},
                        {"role": "user", "content": (
                            f"User emotion: {emotion}\n"
                            f"User message: \"{user_text}\"\n"
                            f"Reply in 1-2 short, warm sentences."
                        )}
                    ],
                    model=self.model,
                    temperature=0.7,
                    max_tokens=150
                )

            return chat_completion.choices[0].message.content.strip()

        except Exception as e:
            print(f"⚠️ Chat Error: {e}")
            return self._fallback_response(is_crisis)

    def detect_crisis(self, text: str) -> tuple[bool, float]:
        """
        Detect if text contains crisis indicators using Groq.
        NOTE: SafetyTriageEngine is the primary crisis detector.
        This is a secondary check you can use optionally.
        Returns: (is_crisis, confidence)
        """
        if not self.client:
            return False, 0.0

        prompt = f"""Check if this message contains mental health crisis indicators (suicidal ideation, self-harm, severe distress): "{text}"
        Return ONLY JSON: {{"is_crisis": true/false, "confidence": 0.9}}
        Do not explain."""

        try:
            chat_completion = self.client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=self.model,
                response_format={"type": "json_object"},
                temperature=0.2,
            )
            
            res = json.loads(chat_completion.choices[0].message.content)
            return res.get('is_crisis', False), float(res.get('confidence', 0.0))
        
        except Exception as e:
            print(f"⚠️ Crisis Detection Error: {e}")
            return False, 0.0

    def _fallback_response(self, is_crisis: bool) -> str:
        """Fallback response when Groq is unavailable"""
        if is_crisis:
            return (
                "I'm deeply concerned about what you've shared. Please reach out for help right now:\n\n"
                "📞 iCall: 9152987821 (24/7, free)\n"
                "📞 AASRA: 9820466726 (24/7, free)\n"
                "📞 Vandrevala: 9999776666 (24/7, free)\n"
                "🚨 Emergency: 112\n\n"
                "You are not alone. Please call one of these numbers now."
            )
        return "I'm here to listen. Please tell me more about how you're feeling."


# Create the singleton instance
ai = GroqService()


# ==================== TEST ====================
if __name__ == "__main__":
    print("\n" + "="*60)
    print("TESTING GROQ SERVICE")
    print("="*60)
    
    test_messages = [
        ("Hi!", False),
        ("Hii!!!", False),
        ("Hello", False),
        ("I'm so sad today", False),
        ("no suicide is my only option", True),
        ("i feel like killing myself", True),
        ("i will commit suicide", True),
        ("Can't take this anymore", False),
    ]
    
    for msg, crisis in test_messages:
        emotion, confidence, meta = ai.detect_emotion(msg)
        response = ai.chat(msg, emotion, crisis)
        
        print(f"\nMessage : '{msg}'")
        print(f"Crisis  : {crisis}")
        print(f"Emotion : {emotion.upper()} ({confidence:.2f})")
        print(f"Response: {response}")
        print("-" * 60)