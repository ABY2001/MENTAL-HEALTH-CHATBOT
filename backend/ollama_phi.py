# ollama_phi.py
# Create this file in your backend directory

import requests
import json
from typing import Optional

class OllamaPhiLLM:
    """Direct Ollama Phi LLM integration"""
    
    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "phi",
        temperature: float = 0.7,
        timeout: int = 10
    ):
        self.base_url = base_url
        self.model = model
        self.temperature = temperature
        self.timeout = timeout
        
        print(f"\n🚀 Initializing Ollama Phi LLM")
        print(f"   Model: {model}")
        print(f"   URL: {base_url}")
        
        self._test_connection()
    
    def _test_connection(self):
        """Test if Ollama is running"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get('models', [])
                model_names = [m['name'] for m in models]
                print(f"✓ Connected to Ollama")
                print(f"  Available models: {model_names}")
                
                if not any(self.model in name for name in model_names):
                    print(f"\n⚠️  WARNING: '{self.model}' not found!")
                    print(f"   Fix: Run in terminal: ollama pull {self.model}")
                else:
                    print(f"✓ Model '{self.model}' is installed")
            else:
                print(f"✗ Ollama returned status {response.status_code}")
        except requests.exceptions.ConnectionError:
            print(f"\n✗ CANNOT CONNECT TO OLLAMA!")
            print(f"   Fix: Make sure Ollama is running")
            print(f"   1. Install from: https://ollama.ai")
            print(f"   2. Open terminal and run: ollama serve")
            print(f"   3. Keep it running while using the API")
        except Exception as e:
            print(f"✗ Connection error: {e}")
    
    def generate(self, prompt: str, max_tokens: int = 200) -> Optional[str]:
        """
        Generate response from Phi
        
        Args:
            prompt: Input text for Phi
            max_tokens: Max response length
            
        Returns:
            Generated text or None if failed
        """
        try:
            print(f"\n🤖 Calling Phi model...")
            print(f"   Prompt: {prompt[:80]}...")
            
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "temperature": self.temperature,
                    "num_predict": max_tokens,
                },
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                result = response.json()
                text = result.get('response', '').strip()
                print(f"✓ Phi response: {text[:80]}...")
                return text
            else:
                print(f"✗ Ollama error {response.status_code}")
                print(f"  Response: {response.text[:200]}")
                return None
                
        except requests.exceptions.Timeout:
            print(f"✗ Phi timeout (10s) - model might be loading, try again")
            return None
        except requests.exceptions.ConnectionError:
            print(f"✗ Cannot connect to Ollama at {self.base_url}")
            print(f"  Is Ollama running? Run: ollama serve")
            return None
        except Exception as e:
            print(f"✗ Error: {e}")
            return None
    
    def is_available(self) -> bool:
        """Check if Ollama server is available"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=2)
            return response.status_code == 200
        except:
            return False