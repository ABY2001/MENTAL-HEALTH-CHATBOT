"""
Safety & Triage Engine - UPDATED
Detects crisis situations, emotional intensity, and triggers appropriate responses
Enhanced with comprehensive keyword detection and pattern matching
"""

import re
from typing import Dict, List, Tuple
from dataclasses import dataclass
from enum import Enum

class RiskLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class EmotionIntensity(Enum):
    MILD = "mild"
    MODERATE = "moderate"
    SEVERE = "severe"
    EXTREME = "extreme"

@dataclass
class SafetyAssessment:
    risk_level: RiskLevel
    crisis_detected: bool
    intensity: EmotionIntensity
    triggered_keywords: List[str]
    recommended_action: str
    crisis_resources: List[Dict[str, str]]
    should_respond_normally: bool
    warning_message: str = ""

class SafetyTriageEngine:
    """
    Evaluates user input for safety risks and emotional intensity
    Enhanced with comprehensive crisis detection
    """
    
    def __init__(self):
        # COMPREHENSIVE CRITICAL KEYWORDS - Updated with all variations
        self.critical_keywords = {
            'suicide': [
                # ⭐ BARE WORD FORMS - must be first
                'suicide', 'suicidal', 'suicidally',
                'commit suicide', 'committing suicide',
                'committed suicide',
                
                # ⭐ "suicide is my only option" type phrases
                'suicide is my only option',
                'suicide is the only option',
                'only option is suicide',
                'suicide is the only way',
                'only way out is suicide',
                'suicide is the answer',
                'suicide is the solution',
                'suicide seems like the only',
                'thinking about suicide',
                'thought about suicide',
                'thoughts of suicide',
                'considering suicide',
                'attempt suicide',
                'attempting suicide',
                'will commit suicide',
                'going to commit suicide',
                'plan to commit suicide',
                'planning to commit suicide',
                'want to commit suicide',
                'suicide tonight',
                'suicide tomorrow',
                'suicide note',
                'suicide attempt',
                
                # Kill myself (all variations)
                'kill myself', 'killing myself', 'killed myself',
                'want to kill myself', 'will kill myself',
                'think about killing myself', 'thinking of killing myself',
                'feel like killing myself', 'feel like i should kill',
                'i am going to kill myself', 'i will kill myself',
                'thinking about killing myself', 'thinking of killing',
                
                # End life
                'end my life', 'end my self', 'end it all',
                'ending my life', 'going to end my life',
                'end it', 'end everything', 'put an end to it',
                
                # Death wish
                'want to die', 'want to be dead', 'wish i was dead',
                'wish i was never born', 'wish i could die',
                'i will die', 'i will be dead', 'i want to die',
                'better off dead', 'better if i was dead',
                'better if i was gone', 'better if i wasn\'t here',
                'better off gone', 'world would be better without me',
                
                # No reason to live
                'no reason to live', 'no point in living', 'pointless to live',
                'no reason to continue', 'can\'t continue',
                'no point to living', 'pointless existence', 'life is pointless',
                
                # Hopelessness and despair
                'no hope left', 'hopeless', 'can\'t go on',
                'can\'t take it anymore', 'can\'t take this',
                'cannot go on', 'cannot continue', 'i give up',
                'hopeless situation', 'nothing matters',
                
                # Burden to others
                'burden to everyone', 'burden to my family',
                'everyone would be better off without me',
                'no one needs me', 'no one cares', 'everyone hates me',
                'i\'m a burden', 'burden on my family',
                
                # Final messages and goodbyes
                'goodbye forever', 'final goodbye', 'final message',
                'this is my last', 'last time talking',
                'never going to see you again', 'goodbye cruel world',
                'this is it', 'goodbye everyone', 'last goodbye',
                'farewell', 'this is the end',
                
                # Specific plans/timeline
                'kill myself tonight', 'kill myself tomorrow',
                'kill myself this week', 'kill myself soon',
                'ending tonight', 'dying tonight', 'die tonight',
                'end it tonight', 'dying tomorrow', 'die tomorrow',
                
                # Preparation and planning
                'preparing to die', 'ready to die',
                'made a plan', 'have a plan to kill',
                'already decided', 'already planned',
                'have decided to die', 'committed to dying',
                'set on dying', 'plan is set',
                
                # Methods and definite planning
                'plan to die', 'planning to die',
                'going to die', 'planning to end it',
                'not going to survive', 'will not survive',
                'will not make it', 'won\'t make it',
                'not going to make it', 'can\'t make it',
                'won\'t survive', 'not survive',
                
                # Active suicide preparation
                'saying goodbye', 'making arrangements',
                'getting affairs in order', 'last will',
                'final arrangements', 'goodbye note',
                'leave a note'
            ],
            
            'self_harm': [
                'cut myself', 'cutting myself', 'want to cut',
                'urge to cut', 'i cut myself', 'keep cutting',
                'cutting my wrists', 'cut my wrists',
                'want to cut myself', 'feel like cutting',
                'cutting urge', 'cuts on my body', 'cutting myself blood',
                'harm myself', 'harming myself', 'hurt myself',
                'hurting myself', 'want to hurt', 'urge to hurt',
                'i hurt myself', 'keep hurting',
                'want to hurt myself', 'hurting urge',
                'burn myself', 'burning myself', 'want to burn',
                'burned myself', 'burn my self', 'self immolation',
                'burning urge', 'want to burn',
                'punch myself', 'hit myself', 'bang my head',
                'punching myself', 'hitting myself', 'head banging',
                'want to punch', 'want to hit',
                'self harm', 'self-harm', 'mutilate',
                'pick at myself', 'scratch myself',
                'starve myself', 'harm self',
                'injure myself', 'injuring myself',
                'scratching myself', 'picking myself',
                'pinch myself hard', 'self injury'
            ],
            
            'overdose': [
                'overdose', 'overdosing', 'overdosed',
                'take all pills', 'take too many pills',
                'pills to die', 'lethal dose',
                'drug overdose', 'will overdose',
                'planning to overdose', 'going to overdose',
                'overdose on pills', 'overdose on drugs',
                'want to overdose', 'take my life with pills',
                'overdose intent', 'lethal overdose',
                'pills overdose', 'drug overdose plan'
            ],
            
            'immediate_danger': [
                'going to kill', 'plan to die', 'planning to die',
                'going to die', 'planning to end it',
                'about to die', 'about to kill myself',
                'about to end it', 'about to go',
                'not going to survive', 'will not survive',
                'will not make it', 'won\'t make it',
                'not going to make it', 'can\'t make it',
                'won\'t survive', 'not survive', 'won\'t live',
                'already decided to', 'have decided to die',
                'committed to dying', 'set on dying',
                'determined to die', 'resolved to die',
                'goodbye forever', 'this is goodbye',
                'final goodbye', 'final message',
                'last time', 'last goodbye',
                'end of the line', 'no way out',
                'last message', 'goodbye notes',
                # ⭐ NEW
                'commit suicide',
                'will commit suicide',
                'going to commit suicide',
                'plan to commit suicide',
                'suicide tonight',
                'suicide tomorrow',
            ]
        }
        
        # HIGH RISK KEYWORDS
        self.high_risk_keywords = {
            'suicidal_ideation': [
                'suicidal thoughts', 'thinking about death', 
                'wish i was dead', 'life not worth living',
                'suicidal ideation', 'death ideation',
                'thinking of ending it', 'considering death',
                'passive death wish', 'death seems appealing',
                # ⭐ NEW
                'only option', 'no other option', 'no other choice',
                'only way out', 'only solution',
            ],
            'severe_depression': [
                'cannot go on', 'give up', 'no hope', 'hopeless',
                'worthless', 'burden to everyone', 'pointless',
                'can\'t take anymore', 'overwhelming', 'depths of despair',
                'deep depression', 'severe depression', 'can\'t cope'
            ],
            'severe_anxiety': [
                'panic attack', 'cannot breathe', 'heart racing',
                'going crazy', 'losing control', 'terror',
                'severe panic', 'having breakdown', 'completely panicked',
                'can\'t breathe properly', 'think i\'m dying'
            ],
            'psychosis': [
                'hearing voices', 'people following', 'conspiracy',
                'they are after me', 'not real', 'hallucinating',
                'seeing things', 'being watched', 'they\'re out to get me'
            ],
            'trauma': [
                'traumatic', 'flashback', 'nightmare', 'abuse', 'assault',
                'ptsd', 'triggered', 'trauma response'
            ]
        }
 
        # MEDIUM RISK KEYWORDS
        self.medium_risk_keywords = {
            'depression': [
                'depressed', 'sad', 'empty', 'numb', 'lonely', 'isolated',
                'feeling down', 'blue', 'down mood', 'depressive'
            ],
            'anxiety': [
                'anxious', 'worried', 'nervous', 'stressed', 'overwhelmed',
                'anxious thoughts', 'worried thoughts', 'stress'
            ],
            'anger': [
                'furious', 'rage', 'angry', 'hate', 'violent thoughts',
                'very angry', 'so mad', 'boiling inside'
            ],
            'trauma': [
                'traumatic', 'flashback', 'nightmare', 'abuse', 'assault'
            ]
        }
        
        # Crisis hotlines and resources (India focused)
        self.crisis_resources = {
            'suicide': [
                {
                    'name': 'AASRA - Lifeline for Suicide Prevention',
                    'number': '9820466726',
                    'whatsapp': '+91 9820466726',
                    'email': 'aasra@aasra.info',
                    'website': 'www.aasra.info',
                    'available': '24/7',
                    'languages': ['Hindi', 'Marathi', 'English'],
                    'description': 'Free and confidential emotional support for suicide prevention'
                },
                {
                    'name': 'iCall - Mental Health Crisis Helpline',
                    'number': '9152987821',
                    'whatsapp': '+91 9152987821',
                    'email': 'info@icallhelpline.org',
                    'website': 'www.icallhelpline.org',
                    'available': '24/7',
                    'languages': ['Hindi', 'English', 'Marathi'],
                    'description': 'Free emotional support and crisis intervention'
                },
                {
                    'name': 'Vandrevala Foundation - Lifeline',
                    'number': '9999 77 6666',
                    'whatsapp': '+91 9999776666',
                    'email': 'help@vandrevalafoundation.com',
                    'website': 'www.vandrevalafoundation.com',
                    'available': '24/7',
                    'languages': ['Hindi', 'Marathi', 'English'],
                    'description': 'Mental health support and suicide prevention'
                },
                {
                    'name': 'RailTel 24x7 Helpline',
                    'number': '182',
                    'toll_free': 'Yes',
                    'available': '24/7',
                    'languages': ['Hindi', 'English', 'Regional'],
                    'description': 'Emergency support at railway stations (ACSM Suicide Prevention)'
                }
            ],
            
            'self_harm': [
                {
                    'name': 'iCall - Self-Harm Support',
                    'number': '9152987821',
                    'whatsapp': '+91 9152987821',
                    'available': '24/7',
                    'languages': ['Hindi', 'English', 'Marathi'],
                    'description': 'Support for self-harm urges and impulses'
                },
                {
                    'name': 'Vandrevala Foundation - Behavioral Health',
                    'number': '9999 77 6666',
                    'whatsapp': '+91 9999776666',
                    'available': '24/7',
                    'languages': ['Hindi', 'Marathi', 'English'],
                    'description': 'Support for self-harm and destructive behaviors'
                },
                {
                    'name': 'AASRA - Emotional Support',
                    'number': '9820466726',
                    'whatsapp': '+91 9820466726',
                    'available': '24/7',
                    'languages': ['Hindi', 'Marathi', 'English'],
                    'description': 'Counseling for self-harm and emotional distress'
                }
            ],
            
            'mental_health': [
                {
                    'name': 'AASRA - Mental Health Counseling',
                    'number': '9820466726',
                    'whatsapp': '+91 9820466726',
                    'email': 'aasra@aasra.info',
                    'available': '24/7',
                    'languages': ['Hindi', 'Marathi', 'English'],
                    'description': 'Mental health counseling, depression, anxiety support'
                },
                {
                    'name': 'iCall - Mental Health Support',
                    'number': '9152987821',
                    'whatsapp': '+91 9152987821',
                    'email': 'info@icallhelpline.org',
                    'available': '24/7',
                    'languages': ['Hindi', 'English', 'Marathi'],
                    'description': 'Support for depression, anxiety, stress management'
                },
                {
                    'name': 'NCPEDP - Mental Health Resources',
                    'number': '011-4141-7800',
                    'email': 'info@ncpedp.org',
                    'website': 'www.ncpedp.org',
                    'available': 'Mon-Fri 9am-6pm IST',
                    'languages': ['Hindi', 'English'],
                    'description': 'Mental health resources and disability support'
                },
                {
                    'name': 'Mental Health Foundation (India)',
                    'number': '1800-425-33-33',
                    'toll_free': 'Yes',
                    'email': 'info@mhfi.org',
                    'website': 'www.mhfi.org',
                    'available': '9am-6pm IST (Mon-Fri)',
                    'languages': ['Hindi', 'English'],
                    'description': 'Mental health information, support, and resources'
                },
                {
                    'name': 'Vandrevala Foundation - General Support',
                    'number': '9999 77 6666',
                    'whatsapp': '+91 9999776666',
                    'available': '24/7',
                    'languages': ['Hindi', 'Marathi', 'English'],
                    'description': 'Mental health awareness and support services'
                }
            ],
            
            'depression': [
                {
                    'name': 'iCall - Depression Support',
                    'number': '9152987821',
                    'whatsapp': '+91 9152987821',
                    'available': '24/7',
                    'languages': ['Hindi', 'English', 'Marathi'],
                    'description': 'Support for depression, low mood, hopelessness'
                },
                {
                    'name': 'Vandrevala Foundation - Depression Help',
                    'number': '9999 77 6666',
                    'whatsapp': '+91 9999776666',
                    'available': '24/7',
                    'languages': ['Hindi', 'Marathi', 'English'],
                    'description': 'Depression counseling and management'
                },
                {
                    'name': 'AASRA - Emotional Support',
                    'number': '9820466726',
                    'whatsapp': '+91 9820466726',
                    'available': '24/7',
                    'languages': ['Hindi', 'Marathi', 'English'],
                    'description': 'Depression and emotional distress support'
                }
            ],
            
            'anxiety': [
                {
                    'name': 'iCall - Anxiety Support',
                    'number': '9152987821',
                    'whatsapp': '+91 9152987821',
                    'available': '24/7',
                    'languages': ['Hindi', 'English', 'Marathi'],
                    'description': 'Support for anxiety, panic attacks, stress'
                },
                {
                    'name': 'Vandrevala Foundation - Anxiety Management',
                    'number': '9999 77 6666',
                    'whatsapp': '+91 9999776666',
                    'available': '24/7',
                    'languages': ['Hindi', 'Marathi', 'English'],
                    'description': 'Anxiety disorders and panic attack support'
                }
            ],
            
            'domestic_violence': [
                {
                    'name': 'National Domestic Violence Hotline (India)',
                    'number': '181',
                    'toll_free': 'Yes',
                    'available': '24/7',
                    'languages': ['Hindi', 'English', 'Regional'],
                    'description': 'Support for domestic violence victims (for women)'
                },
                {
                    'name': 'AASRA - Domestic Violence Support',
                    'number': '9820466726',
                    'whatsapp': '+91 9820466726',
                    'available': '24/7',
                    'languages': ['Hindi', 'Marathi', 'English'],
                    'description': 'Support for domestic violence survivors'
                },
                {
                    'name': 'Sneha - Mumbai Based NGO',
                    'number': '9922004948',
                    'email': 'contact@snehamumbai.org',
                    'available': '24/7',
                    'languages': ['Hindi', 'Marathi', 'English'],
                    'description': 'Support for violence survivors and counseling'
                }
            ],
            
            'substance_abuse': [
                {
                    'name': 'All India Institute of Medical Sciences (AIIMS) - De-addiction',
                    'number': '011-26589169',
                    'email': 'aiimsdrugs@gmail.com',
                    'available': 'Mon-Fri 9am-5pm IST',
                    'languages': ['Hindi', 'English'],
                    'description': 'Professional de-addiction and substance abuse treatment'
                },
                {
                    'name': 'Narcotics Anonymous India',
                    'number': '7738-022022',
                    'whatsapp': '+91 7738022022',
                    'email': 'naindia@outlook.com',
                    'available': '24/7',
                    'languages': ['Hindi', 'English'],
                    'description': 'Support for substance abuse recovery'
                },
                {
                    'name': 'iCall - Substance Abuse Support',
                    'number': '9152987821',
                    'whatsapp': '+91 9152987821',
                    'available': '24/7',
                    'languages': ['Hindi', 'English', 'Marathi'],
                    'description': 'Support for substance abuse and addiction issues'
                }
            ],
            
            'emergency': [
                {
                    'name': 'National Emergency Helpline',
                    'number': '100',
                    'toll_free': 'Yes',
                    'available': '24/7',
                    'languages': ['Hindi', 'English', 'Regional'],
                    'description': 'Police emergency response'
                },
                {
                    'name': 'Ambulance/Medical Emergency',
                    'number': '102',
                    'toll_free': 'Yes',
                    'available': '24/7',
                    'languages': ['Hindi', 'English', 'Regional'],
                    'description': 'Emergency medical services'
                },
                {
                    'name': 'Fire Emergency',
                    'number': '101',
                    'toll_free': 'Yes',
                    'available': '24/7',
                    'languages': ['Hindi', 'English'],
                    'description': 'Fire services and rescue'
                }
            ],
            
            'professional_help': [
                {
                    'name': 'Psychology Foundation of India',
                    'number': '011-4150-2442',
                    'email': 'info@psychologyfoundation.org',
                    'website': 'www.psychologyfoundation.org',
                    'available': 'Mon-Fri 9am-6pm IST',
                    'languages': ['Hindi', 'English'],
                    'description': 'Connect with licensed therapists and psychologists'
                },
                {
                    'name': 'Fortis Mental Health Hospitals',
                    'number': '1800-102-5008',
                    'toll_free': 'Yes',
                    'available': '24/7',
                    'languages': ['Hindi', 'English'],
                    'description': 'Professional psychiatric and mental health care'
                },
                {
                    'name': 'Max Healthcare - Mental Health',
                    'number': '8860-018880',
                    'available': '24/7',
                    'languages': ['Hindi', 'English'],
                    'description': 'Professional mental health services across India'
                },
                {
                    'name': 'Apollo Hospitals - Psychiatry',
                    'number': '1860-500-1066',
                    'toll_free': 'Yes',
                    'available': '24/7',
                    'languages': ['Hindi', 'English'],
                    'description': 'Mental health and psychiatric care'
                }
            ],
            
            'grief_loss': [
                {
                    'name': 'iCall - Grief Support',
                    'number': '9152987821',
                    'whatsapp': '+91 9152987821',
                    'available': '24/7',
                    'languages': ['Hindi', 'English', 'Marathi'],
                    'description': 'Support for grief, loss, and bereavement'
                },
                {
                    'name': 'AASRA - Loss and Grief Counseling',
                    'number': '9820466726',
                    'whatsapp': '+91 9820466726',
                    'available': '24/7',
                    'languages': ['Hindi', 'Marathi', 'English'],
                    'description': 'Counseling for grief and loss'
                }
            ],
            
            'family_issues': [
                {
                    'name': 'iCall - Family Counseling',
                    'number': '9152987821',
                    'whatsapp': '+91 9152987821',
                    'available': '24/7',
                    'languages': ['Hindi', 'English', 'Marathi'],
                    'description': 'Support for family conflicts, relationship issues'
                },
                {
                    'name': 'Vandrevala Foundation - Family Support',
                    'number': '9999 77 6666',
                    'whatsapp': '+91 9999776666',
                    'available': '24/7',
                    'languages': ['Hindi', 'Marathi', 'English'],
                    'description': 'Family counseling and relationship support'
                }
            ]
        }
        
    def evaluate(self, text: str, emotion: str, confidence: float) -> SafetyAssessment:
        """
        Main evaluation method - analyzes text and emotion for safety risks
        """
        text_lower = text.lower()
        
        # Step 1: Check for critical crisis keywords
        critical_detected, critical_triggers = self._check_keywords(
            text_lower, self.critical_keywords
        )
        
        # Step 2: Check for harmful patterns (catches variations)
        pattern_critical, pattern_triggers = self._check_harmful_patterns(text_lower)
        critical_detected = critical_detected or pattern_critical
        critical_triggers.extend(pattern_triggers)
        
        if critical_detected:
            return self._create_critical_assessment(critical_triggers)
        
        # Step 3: Check for high-risk keywords
        high_risk_detected, high_risk_triggers = self._check_keywords(
            text_lower, self.high_risk_keywords
        )
        
        if high_risk_detected:
            return self._create_high_risk_assessment(high_risk_triggers)
        
        # Step 4: Check for medium-risk keywords
        medium_risk_detected, medium_triggers = self._check_keywords(
            text_lower, self.medium_risk_keywords
        )
        
        # Step 5: Assess emotional intensity
        intensity = self._assess_intensity(emotion, confidence, text_lower)
        
        # Step 6: Combine assessments
        if medium_risk_detected or intensity in [EmotionIntensity.SEVERE, EmotionIntensity.EXTREME]:
            return self._create_medium_risk_assessment(medium_triggers, intensity, emotion)
        
        # Step 7: Low risk - normal conversation
        return self._create_low_risk_assessment(intensity, emotion)
    
    def _check_keywords(self, text: str, keyword_dict: Dict[str, List[str]]) -> Tuple[bool, List[str]]:
        """
        Check if text contains any keywords from the dictionary
        """
        triggered = []
        for category, keywords in keyword_dict.items():
            for keyword in keywords:
                if keyword in text:
                    triggered.append(keyword)
        
        return len(triggered) > 0, triggered
    
    def _check_harmful_patterns(self, text: str) -> Tuple[bool, List[str]]:
        """
        Check for harmful patterns using regex.
        Catches variations and indirect language.
        """
        patterns = [
            # ⭐ SUICIDE CONTEXTUAL PATTERNS (catches "suicide is my only option")
            r'suicide.*(only|option|way|out|answer|solution|choice|left)',
            r'(only|option|way|out|answer|solution).*(suicide)',
            r'(commit|committing|committed|attempting|attempt|plan|planning|consider|considering|thinking).*suicide',
            r'suicide.*(tonight|tomorrow|soon|today|now)',
            r'(will|going to|want to|need to|have to|must).*suicide',
            r'no.*(other)?.*(option|choice|way|reason).*(left|out|to live)?',
            
            # Kill myself patterns
            r'(kill|end|harm|hurt|cut|burn|stab).*myself',
            r'(suicid|kill|die|death|end).*myself',
            r'want.*to.*(die|kill|end|harm)',
            r'think.*about.*(kill|die|end|harm)',
            r'feel.*like.*(kill|die|end|harm)',
            r'plan.*to.*(die|kill|end)',
            r'ready.*to.*(die|kill)',
            r'better.*off.*(dead|gone)',
            r'(can\'t|cannot).*go.*on',
            r'(no|without).*(point|reason|hope).*(in )?(living|life|going on)',
            r'burden.*to.*(everyone|family|people|others)',
            r'everyone.*(would be |is )better.*(without|off without).*me',
            r'(always|will|going to).*(die|kill)',
            r'(only|just).*way.*out',
            r'no.*reason.*(to )?live',
            r'life.*not.*worth',
            r'(going|about).*to.*(kill|die|end)',
            r'(final|last).*(message|goodbye|words)',
            r'not.*going.*survive',
            r'decided.*to.*die',
            r'(cutting|cut).*wrist',
            r'hurt.*myself.*badly',
            
            # "I will commit suicide" type
            r'(i will|i\'m going to|i am going to|i\'ll).*suicide',
            r'(i will|i\'m going to|i am going to|i\'ll).*(kill|end|harm).*myself',
        ]
        
        triggered = []
        for pattern in patterns:
            if re.search(pattern, text, re.IGNORECASE):
                triggered.append(f"pattern:{pattern}")
        
        return len(triggered) > 0, triggered
    
    def _assess_intensity(self, emotion: str, confidence: float, text: str) -> EmotionIntensity:
        """
        Assess emotional intensity based on emotion type, confidence, and text patterns
        """
        extreme_words = ['extremely', 'unbearably', 'completely', 'totally', 'absolutely', 'can\'t take']
        severe_words = ['very', 'really', 'so', 'too', 'incredibly', 'awful', 'terrible']
        
        has_extreme = any(word in text for word in extreme_words)
        has_severe = any(word in text for word in severe_words)
        
        intense_emotions = ['angry', 'fearful', 'sad']
        
        if has_extreme or (emotion in intense_emotions and confidence > 0.85):
            return EmotionIntensity.EXTREME
        elif has_severe or (emotion in intense_emotions and confidence > 0.70):
            return EmotionIntensity.SEVERE
        elif emotion in intense_emotions or confidence > 0.60:
            return EmotionIntensity.MODERATE
        else:
            return EmotionIntensity.MILD
    
    def _create_critical_assessment(self, triggers: List[str]) -> SafetyAssessment:
        """Create assessment for critical risk situations"""
        return SafetyAssessment(
            risk_level=RiskLevel.CRITICAL,
            crisis_detected=True,
            intensity=EmotionIntensity.EXTREME,
            triggered_keywords=triggers,
            recommended_action="IMMEDIATE_INTERVENTION",
            crisis_resources=self.crisis_resources['suicide'],
            should_respond_normally=False,
            warning_message="I'm very concerned about your safety. Please reach out to a crisis helpline immediately."
        )
    
    def _create_high_risk_assessment(self, triggers: List[str]) -> SafetyAssessment:
        """Create assessment for high-risk situations"""
        return SafetyAssessment(
            risk_level=RiskLevel.HIGH,
            crisis_detected=True,
            intensity=EmotionIntensity.EXTREME,
            triggered_keywords=triggers,
            recommended_action="URGENT_SUPPORT",
            crisis_resources=self.crisis_resources['mental_health'],
            should_respond_normally=True,
            warning_message="I'm concerned about what you're going through. Please consider speaking with a mental health professional."
        )
    
    def _create_medium_risk_assessment(
        self, triggers: List[str], intensity: EmotionIntensity, emotion: str
    ) -> SafetyAssessment:
        """Create assessment for medium-risk situations"""
        return SafetyAssessment(
            risk_level=RiskLevel.MEDIUM,
            crisis_detected=False,
            intensity=intensity,
            triggered_keywords=triggers,
            recommended_action="MONITOR_AND_SUPPORT",
            crisis_resources=self.crisis_resources['mental_health'],
            should_respond_normally=True,
            warning_message=""
        )
    
    def _create_low_risk_assessment(
        self, intensity: EmotionIntensity, emotion: str
    ) -> SafetyAssessment:
        """Create assessment for low-risk situations"""
        return SafetyAssessment(
            risk_level=RiskLevel.LOW,
            crisis_detected=False,
            intensity=intensity,
            triggered_keywords=[],
            recommended_action="CONTINUE_CONVERSATION",
            crisis_resources=[],
            should_respond_normally=True,
            warning_message=""
        )


# ==================== TEST ====================
if __name__ == "__main__":
    engine = SafetyTriageEngine()
    
    test_cases = [
        # ⭐ THE FAILING CASE FROM YOUR SCREENSHOT
        ("no suicide is my only option", "sad", 0.9),
        ("suicide is my only option", "sad", 0.9),
        ("i think suicide is the only way", "sad", 0.85),
        
        # Original test cases
        ("feel like killing myself", "sad", 0.9),
        ("killing myself", "sad", 0.85),
        ("want to kill myself", "angry", 0.8),
        ("thinking about killing myself", "sad", 0.9),
        ("i will kill myself tonight", "sad", 0.95),
        ("have a plan to kill myself", "fearful", 0.9),
        ("cutting myself", "angry", 0.7),
        ("burning myself", "sad", 0.8),
        ("overdosing tonight", "sad", 0.95),
        ("better off dead", "sad", 0.85),
        ("no reason to live", "sad", 0.9),
        ("burden to everyone", "sad", 0.8),
        ("i will commit suicide", "sad", 0.95),
        ("commit suicide", "sad", 0.9),
    ]
    
    print("=" * 80)
    print("SAFETY ENGINE TEST RESULTS")
    print("=" * 80)
    
    passed = 0
    failed = 0
    
    for test_text, emotion, confidence in test_cases:
        result = engine.evaluate(test_text, emotion, confidence)
        is_critical = result.risk_level in [RiskLevel.CRITICAL, RiskLevel.HIGH]
        status = "✅ PASS" if is_critical else "❌ FAIL"
        
        if is_critical:
            passed += 1
        else:
            failed += 1
        
        print(f"\n{status} | Input: '{test_text}'")
        print(f"       Risk: {result.risk_level.value} | Crisis: {result.crisis_detected} | Action: {result.recommended_action}")
        print("-" * 80)
    
    print(f"\n📊 Results: {passed} passed, {failed} failed out of {len(test_cases)} tests")