import os
import json
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

class GroqService:
    def __init__(self):
        self.api_key = os.getenv("GROQ_API_KEY")
        self.client = None
        
        # ⭐ UPDATED MODEL: Llama 3.1 (Fastest & Current)
        # Old 'llama3-8b-8192' is decommissioned.
        self.model = "llama-3.1-8b-instant"
        
        if not self.api_key:
            print("❌ GROQ ERROR: No GROQ_API_KEY found in .env")
        else:
            try:
                self.client = Groq(api_key=self.api_key)
                # Quick connection test
                self.client.models.list()
                print(f"✓ Groq Service Online (Model: {self.model})")
            except Exception as e:
                print(f"❌ Groq Connection Failed: {e}")
                self.client = None

    def detect_emotion(self, text: str):
        if not self.client: return "neutral", 0.0, {}

        prompt = f"""Analyze the emotion of this text: "{text}"
        Return ONLY valid JSON: {{ "emotion": "happy|sad|angry|neutral|fearful|calm", "confidence": 0.9 }}
        Do not explain. Only return JSON."""

        try:
            chat_completion = self.client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=self.model,
                # ⭐ JSON MODE (Guarantees valid JSON format)
                response_format={"type": "json_object"},
                temperature=0.2,
            )
            
            content = chat_completion.choices[0].message.content
            res = json.loads(content)
            return res.get('emotion', 'neutral').lower(), float(res.get('confidence', 0.5)), {}
        
        except Exception as e:
            print(f"⚠️ Emotion Error: {e}")
            return "neutral", 0.0, {}

    def chat(self, user_text: str, emotion: str, is_crisis: bool):
        if not self.client: return "I'm here to listen."

        role = "Crisis Counselor" if is_crisis else "Supportive Friend"
        prompt = f"""Role: {role} | User Emotion: {emotion}
        User Message: "{user_text}"
        Reply in 1-2 short, warm sentences."""

        try:
            chat_completion = self.client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=self.model,
                temperature=0.7,
                max_tokens=150
            )
            return chat_completion.choices[0].message.content.strip()
        except Exception as e:
            print(f"⚠️ Chat Error: {e}")
            return "I'm here to listen."

# Create the singleton instance
ai = GroqService()