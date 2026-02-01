# improved_emotion_detection.py
# Better emotion detection with proper negation and context handling

from typing import Tuple, Dict
import re

class ImprovedEmotionDetector:
    """
    Smart emotion detection that handles:
    - Negation context ("don't have a good grade" = sad, not happy)
    - Sentence-level analysis instead of just word matching
    - Multiple sentences
    - Intensifiers and modifiers
    """
    
    def __init__(self):
        """Initialize with better keyword database"""
        
        # Keywords ONLY for positive contexts (not negated)
        self.happy_keywords = [
            'happy', 'joy', 'joyful', 'wonderful', 'excited', 'excellent', 'amazing',
            'fantastic', 'delighted', 'cheerful', 'blessed', 'love', 'awesome',
            'brilliant', 'contented', 'satisfied', 'hopeful', 'optimistic', 'proud',
            'grateful', 'thrilled', 'elated', 'overjoyed', 'ecstatic'
        ]
        
        self.sad_keywords = [
            'sad', 'depressed', 'down', 'unhappy', 'miserable', 'terrible', 'awful',
            'cry', 'crying', 'upset', 'disappointed', 'hurt', 'lonely', 'hopeless',
            'worthless', 'devastated', 'empty', 'broken', 'heartbroken', 'guilty',
            'ashamed', 'regret', 'loss', 'grief', 'blue', 'gloomy', 'despair',
            'melancholy', 'sorrowful', 'disheartened', 'devastated', 'defeated'
        ]
        
        self.angry_keywords = [
            'angry', 'mad', 'furious', 'annoyed', 'irritated', 'frustrated', 'rage',
            'hate', 'pissed', 'outraged', 'bitter', 'resentful', 'enraged', 'hostile',
            'aggressive', 'infuriating', 'unfair', 'livid', 'incensed', 'exasperated',
            'vexed', 'cross', 'irate', 'hectic', 'chaotic', 'overwhelming', 'stressful',
            'draining', 'exhausting'
        ]
        
        self.fearful_keywords = [
            'scared', 'afraid', 'anxious', 'worried', 'nervous', 'fear', 'panic',
            'terrified', 'frightened', 'uneasy', 'stressed', 'tense', 'threatened',
            'insecure', 'unsafe', 'paranoid', 'dread', 'pressure', 'apprehensive',
            'distressed', 'anguished', 'concerned', 'alarmed', 'petrified', 'concern'
        ]
        
        self.calm_keywords = [
            'calm', 'peaceful', 'relaxed', 'serene', 'tranquil', 'composed', 'content',
            'comfortable', 'zen', 'mellow', 'balanced', 'stable', 'quiet', 'still',
            'settled', 'peace', 'okay', 'fine', 'alright'
        ]
        
        self.neutral_keywords = [
            'okay', 'fine', 'alright', 'normal', 'usual', 'average', 'ordinary',
            'regular', 'standard', 'meh'
        ]
        
        self.surprised_keywords = [
            'surprised', 'shocked', 'amazed', 'astonished', 'unexpected', 'wow',
            'incredible', 'unbelievable', 'suddenly', 'didn\'t expect', 'stunned',
            'flabbergasted', 'bewildered', 'taken aback'
        ]
        
        self.disgust_keywords = [
            'disgusted', 'revolted', 'gross', 'nasty', 'sick', 'repulsive', 'creepy',
            'disturbing', 'yuck', 'horrible', 'dirty', 'disgusting', 'abhorrent',
            'loathsome', 'vile'
        ]
        
        # Negation patterns
        self.negation_patterns = [
            r'\b(not|no|never|isn\'t|aren\'t|wasn\'t|weren\'t|don\'t|doesn\'t|didn\'t|haven\'t|hasn\'t|wouldn\'t|shouldn\'t|couldn\'t)\b',
            r'\bdon\'t\s+have',
            r'\b(lack|missing|without|fail)',
        ]
        
        # Intensifiers
        self.intensifiers = ['very', 'really', 'so', 'extremely', 'absolutely', 'totally', 'completely', 'utterly']
    
    def detect_emotion(self, text: str) -> Tuple[str, float, Dict]:
        """
        Detect emotion from text with proper context awareness
        """
        text_lower = text.lower()
        text_lower = re.sub(r'\s+', ' ', text_lower).strip()
        
        print(f"\n🔍 Analyzing: '{text}'")
        
        # Split into sentences for better context
        sentences = re.split(r'[.!?]+', text_lower)
        
        emotion_scores = {
            'happy': 0, 'sad': 0, 'angry': 0, 'fearful': 0,
            'calm': 0, 'neutral': 0, 'surprised': 0, 'disgust': 0
        }
        
        # Analyze each sentence separately
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            sentence_emotions = self._analyze_sentence(sentence)
            
            # Add sentence scores to total
            for emotion, score in sentence_emotions.items():
                emotion_scores[emotion] += score
        
        print(f"📊 Scores: {emotion_scores}")
        
        # Find dominant emotion
        detected_emotion = max(emotion_scores, key=emotion_scores.get)
        max_score = emotion_scores[detected_emotion]
        
        # If no emotion found, default to neutral
        if max_score <= 0:
            detected_emotion = 'neutral'
            confidence = 0.5
            print(f"   → No clear emotion, defaulting to 'neutral'")
        else:
            # Calculate confidence
            total_score = sum(max(s, 0) for s in emotion_scores.values()) or 1
            confidence = max_score / total_score
            confidence = min(confidence, 0.95)
        
        # Normalize emotions to probabilities
        total_positive = sum(max(s, 0) for s in emotion_scores.values()) or 1
        all_emotions = {
            emotion: max(0, emotion_scores[emotion]) / total_positive
            for emotion in emotion_scores
        }
        
        print(f"✅ Detected: {detected_emotion.upper()} (confidence: {confidence:.2f})")
        
        return detected_emotion, confidence, all_emotions
    
    def _analyze_sentence(self, sentence: str) -> Dict[str, float]:
        """Analyze a single sentence"""
        scores = {
            'happy': 0, 'sad': 0, 'angry': 0, 'fearful': 0,
            'calm': 0, 'neutral': 0, 'surprised': 0, 'disgust': 0
        }
        
        if not sentence.strip():
            return scores
        
        # Check if sentence is negated
        is_negated = self._is_negated(sentence)
        
        # Check for intensifiers
        is_intensified = any(intensifier in sentence for intensifier in self.intensifiers)
        
        # Score each emotion
        happy_matches = sum(1 for word in self.happy_keywords if re.search(r'\b' + word + r'\b', sentence))
        sad_matches = sum(1 for word in self.sad_keywords if re.search(r'\b' + word + r'\b', sentence))
        angry_matches = sum(1 for word in self.angry_keywords if re.search(r'\b' + word + r'\b', sentence))
        fearful_matches = sum(1 for word in self.fearful_keywords if re.search(r'\b' + word + r'\b', sentence))
        calm_matches = sum(1 for word in self.calm_keywords if re.search(r'\b' + word + r'\b', sentence))
        surprised_matches = sum(1 for word in self.surprised_keywords if re.search(r'\b' + word + r'\b', sentence))
        disgust_matches = sum(1 for word in self.disgust_keywords if re.search(r'\b' + word + r'\b', sentence))
        
        # If negated, flip happy to sad and apply negative weight
        if is_negated:
            print(f"   ✗ Sentence is NEGATED: '{sentence[:60]}...'")
            
            if happy_matches > 0:
                print(f"      - Found 'good/excellent' but negated → flip to SAD")
                scores['sad'] += happy_matches * 1.3
            
            if calm_matches > 0:
                scores['sad'] += calm_matches * 1.2
        else:
            # Normal scoring
            if happy_matches > 0:
                print(f"   ✓ Happy: {happy_matches} matches")
                scores['happy'] += happy_matches * 1.0
            
            if calm_matches > 0:
                scores['calm'] += calm_matches * 0.9
        
        if sad_matches > 0:
            print(f"   ✓ Sad: {sad_matches} matches")
            scores['sad'] += sad_matches * 1.3
        
        if angry_matches > 0:
            print(f"   ✓ Angry: {angry_matches} matches")
            scores['angry'] += angry_matches * 1.2
        
        if fearful_matches > 0:
            print(f"   ✓ Fearful: {fearful_matches} matches")
            scores['fearful'] += fearful_matches * 1.3
        
        if surprised_matches > 0:
            print(f"   ✓ Surprised: {surprised_matches} matches")
            scores['surprised'] += surprised_matches * 0.9
        
        if disgust_matches > 0:
            print(f"   ✓ Disgust: {disgust_matches} matches")
            scores['disgust'] += disgust_matches * 1.0
        
        # Apply intensifier
        if is_intensified and max(scores.values()) > 0:
            dominant = max(scores, key=scores.get)
            scores[dominant] *= 1.3
            print(f"   ↑ Intensified ({dominant})")
        
        return scores
    
    def _is_negated(self, sentence: str) -> bool:
        """Check if sentence contains negation"""
        for pattern in self.negation_patterns:
            if re.search(pattern, sentence):
                return True
        return False


# ==================== TEST ====================
if __name__ == "__main__":
    detector = ImprovedEmotionDetector()
    
    test_cases = [
        "Today was a hectic day",
        "I'm feeling great!",
        "Not happy at all",
        "I'm very sad",
        "Everything is fine",
        "I'm scared and nervous",
        "This is disgusting!",
        "Wow! I wasn't expecting this!",
        "I'm so angry right now",
        "my third semester results came out i dont have a good grade",  # Your test case
        "I got excellent grades!",
        "I failed my exam",
    ]
    
    for text in test_cases:
        emotion, confidence, _ = detector.detect_emotion(text)
        print(f"Result: {emotion} ({confidence:.2f})\n")