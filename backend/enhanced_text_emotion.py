"""
Enhanced Text-Based Emotion Detection
Uses intent patterns and better keyword matching for accurate emotion detection
"""

import json
import re
from typing import Tuple, Dict
from collections import defaultdict

class EnhancedTextEmotionDetector:
    """
    Enhanced emotion detector using intent patterns and contextual analysis
    """
    
    def __init__(self, intents_file: str = "intents.json"):
        """
        Initialize with intent patterns from JSON file
        
        Args:
            intents_file: Path to intents.json file
        """
        self.emotion_patterns = self._load_intents(intents_file)
        self._build_keyword_index()
    
    def _load_intents(self, filepath: str) -> Dict:
        """Load and organize intents by emotion/tag"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Organize intents by emotion tags
            emotion_intents = defaultdict(list)
            
            for intent in data.get('intents', []):
                tag = intent.get('tag', '')
                patterns = intent.get('patterns', [])
                
                # Map tags to emotions
                emotion = self._tag_to_emotion(tag)
                if emotion:
                    emotion_intents[emotion].extend(patterns)
            
            print(f"✓ Loaded {len(emotion_intents)} emotion categories from intents")
            return dict(emotion_intents)
            
        except FileNotFoundError:
            print(f"⚠️ Intents file not found: {filepath}")
            return self._get_default_patterns()
        except Exception as e:
            print(f"⚠️ Error loading intents: {e}")
            return self._get_default_patterns()
    
    def _tag_to_emotion(self, tag: str) -> str:
        """Map intent tags to emotion categories"""
        emotion_mapping = {
            # Negative emotions
            'sad': 'sad',
            'depressed': 'sad',
            'worthless': 'sad',
            'death': 'sad',
            'lonely': 'sad',
            'sleep': 'sad',
            
            # Anxiety/Fear
            'anxious': 'fearful',
            'stressed': 'fearful',
            'scared': 'fearful',
            'suicide': 'fearful',  # Critical - treated as high fear/crisis
            
            # Anger
            'hate-you': 'angry',
            'hate-me': 'angry',
            
            # Positive
            'happy': 'happy',
            'thanks': 'happy',
            'greeting': 'neutral',
            
            # Neutral
            'casual': 'neutral',
            'about': 'neutral',
            'help': 'neutral',
        }
        
        return emotion_mapping.get(tag.lower(), None)
    
    def _get_default_patterns(self) -> Dict:
        """Fallback patterns if intents.json not available"""
        return {
            'sad': [
                'sad', 'depressed', 'lonely', 'worthless', 'hopeless',
                'crying', 'down', 'miserable', 'empty', 'numb'
            ],
            'fearful': [
                'anxious', 'worried', 'scared', 'stressed', 'nervous',
                'panic', 'afraid', 'terrified', 'overwhelmed'
            ],
            'angry': [
                'angry', 'mad', 'furious', 'hate', 'frustrated',
                'annoyed', 'irritated', 'rage'
            ],
            'happy': [
                'happy', 'great', 'wonderful', 'good', 'excellent',
                'amazing', 'fantastic', 'joy', 'cheerful'
            ]
        }
    
    def _build_keyword_index(self):
        """Build optimized keyword search index"""
        self.keyword_index = {}
        
        for emotion, patterns in self.emotion_patterns.items():
            for pattern in patterns:
                # Convert pattern to lowercase and extract keywords
                words = re.findall(r'\w+', pattern.lower())
                for word in words:
                    if len(word) > 2:  # Ignore very short words
                        if word not in self.keyword_index:
                            self.keyword_index[word] = {}
                        
                        if emotion not in self.keyword_index[word]:
                            self.keyword_index[word][emotion] = 0
                        
                        self.keyword_index[word][emotion] += 1
    
    def detect_emotion(self, text: str) -> Tuple[str, float, Dict[str, float]]:
        """
        Detect emotion from text with improved accuracy
        
        Args:
            text: Input text from user
        
        Returns:
            (emotion, confidence, all_emotions_dict)
        """
        text_lower = text.lower()
        
        # Step 1: Check for exact pattern matches (highest priority)
        for emotion, patterns in self.emotion_patterns.items():
            for pattern in patterns:
                if pattern.lower() in text_lower:
                    # Exact match - high confidence
                    return emotion, 0.95, self._build_emotion_dict(emotion, 0.95)
        
        # Step 2: Keyword-based scoring with context
        emotion_scores = defaultdict(float)
        words = re.findall(r'\w+', text_lower)
        total_matches = 0
        
        for word in words:
            if word in self.keyword_index:
                for emotion, count in self.keyword_index[word].items():
                    # Weight by keyword frequency in patterns
                    emotion_scores[emotion] += count * 1.5
                    total_matches += 1
        
        # Step 3: Contextual modifiers
        # Negative indicators
        negative_words = ['not', 'no', 'never', "don't", "can't", "won't"]
        has_negative = any(neg in text_lower for neg in negative_words)
        
        # Intensity modifiers
        intense_words = ['very', 'extremely', 'really', 'so', 'too']
        intensity_multiplier = 1.0
        for intense in intense_words:
            if intense in text_lower:
                intensity_multiplier = 1.3
                break
        
        # Apply modifiers
        for emotion in emotion_scores:
            emotion_scores[emotion] *= intensity_multiplier
        
        # Step 4: Specific phrase detection
        specific_checks = {
            'my marks are not': 'sad',
            'exam result': 'fearful',
            'not good': 'sad',
            'not that good': 'sad',
            'failed': 'sad',
            'disappointed': 'sad',
            'upset': 'sad',
        }
        
        for phrase, emotion in specific_checks.items():
            if phrase in text_lower:
                emotion_scores[emotion] += 10  # High weight for specific phrases
        
        # Step 5: Determine emotion
        if not emotion_scores:
            return 'neutral', 0.5, self._build_emotion_dict('neutral', 0.5)
        
        # Get emotion with highest score
        detected_emotion = max(emotion_scores, key=emotion_scores.get)
        max_score = emotion_scores[detected_emotion]
        
        # Calculate confidence (0.6 to 0.95)
        if max_score > 15:
            confidence = 0.95
        elif max_score > 10:
            confidence = 0.90
        elif max_score > 5:
            confidence = 0.80
        else:
            confidence = min(0.6 + (max_score * 0.05), 0.95)
        
        # Build emotion distribution
        all_emotions = self._build_emotion_dict(detected_emotion, confidence)
        
        return detected_emotion, confidence, all_emotions
    
    def _build_emotion_dict(self, primary_emotion: str, confidence: float) -> Dict[str, float]:
        """Build emotion probability distribution"""
        emotions = ['happy', 'sad', 'angry', 'fearful', 'calm', 'neutral', 'surprised', 'disgust']
        
        all_emotions = {}
        remaining_prob = 1.0 - confidence
        
        for emotion in emotions:
            if emotion == primary_emotion:
                all_emotions[emotion] = confidence
            else:
                all_emotions[emotion] = remaining_prob / (len(emotions) - 1)
        
        return {k: round(v, 3) for k, v in all_emotions.items()}

# Integration function for FastAPI
def detect_emotion_from_text_enhanced(text: str, detector: EnhancedTextEmotionDetector) -> Tuple[str, float, Dict]:
    """
    Enhanced emotion detection for FastAPI integration
    """
    return detector.detect_emotion(text)

# Test cases
if __name__ == "__main__":
    # Initialize detector
    detector = EnhancedTextEmotionDetector("intents.json")
    
    test_cases = [
        "today i got exam result my marks are not that good",
        "I'm feeling very happy today",
        "I'm so depressed I can't take it anymore",
        "I'm really anxious about my exams",
        "I'm so angry at everyone",
        "I feel okay, nothing special",
        "I want to kill myself",
        "I failed my exam and feel terrible",
        "My exam results were amazing!",
        "I'm worried about my future"
    ]
    
    print("\n" + "="*70)
    print("ENHANCED EMOTION DETECTION - TEST CASES")
    print("="*70)
    
    for text in test_cases:
        emotion, confidence, all_emotions = detector.detect_emotion(text)
        
        print(f"\nText: '{text}'")
        print(f"Emotion: {emotion}")
        print(f"Confidence: {confidence:.2%}")
        print(f"Top 3: {dict(sorted(all_emotions.items(), key=lambda x: x[1], reverse=True)[:3])}")
        print("-"*70)