"""
Safety & Triage Engine
Detects crisis situations, emotional intensity, and triggers appropriate responses
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
    """
    
    def __init__(self):
        # Critical crisis keywords (immediate intervention)
        self.critical_keywords = {
            'suicide': ['suicide', 'kill myself', 'end my life', 'want to die', 
                       'better off dead', 'end it all', 'take my life', 'no reason to live'],
            'self_harm': ['cut myself', 'hurt myself', 'self harm', 'burn myself',
                         'harm myself', 'injure myself', 'mutilate'],
            'overdose': ['overdose', 'take all pills', 'pills to die', 'lethal dose'],
            'immediate_danger': ['going to kill', 'plan to die', 'goodbye forever',
                                'final message', 'ending tonight']
        }
        
        # High-risk keywords (urgent attention needed)
        self.high_risk_keywords = {
            'suicidal_ideation': ['suicidal thoughts', 'thinking about death', 
                                 'wish I was dead', 'life not worth living'],
            'severe_depression': ['cannot go on', 'give up', 'no hope', 'hopeless',
                                 'worthless', 'burden to everyone', 'pointless'],
            'severe_anxiety': ['panic attack', 'cannot breathe', 'heart racing',
                              'going crazy', 'losing control', 'terror'],
            'psychosis': ['hearing voices', 'people following', 'conspiracy',
                         'they are after me', 'not real', 'hallucinating']
        }
        
        # Medium-risk keywords (monitoring needed)
        self.medium_risk_keywords = {
            'depression': ['depressed', 'sad', 'empty', 'numb', 'lonely', 'isolated'],
            'anxiety': ['anxious', 'worried', 'nervous', 'stressed', 'overwhelmed'],
            'anger': ['furious', 'rage', 'angry', 'hate', 'violent thoughts'],
            'trauma': ['traumatic', 'flashback', 'nightmare', 'abuse', 'assault']
        }
        
        # Crisis hotlines and resources
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
        
        if critical_detected:
            return self._create_critical_assessment(critical_triggers)
        
        # Step 2: Check for high-risk keywords
        high_risk_detected, high_risk_triggers = self._check_keywords(
            text_lower, self.high_risk_keywords
        )
        
        if high_risk_detected:
            return self._create_high_risk_assessment(high_risk_triggers)
        
        # Step 3: Check for medium-risk keywords
        medium_risk_detected, medium_triggers = self._check_keywords(
            text_lower, self.medium_risk_keywords
        )
        
        # Step 4: Assess emotional intensity
        intensity = self._assess_intensity(emotion, confidence, text_lower)
        
        # Step 5: Combine assessments
        if medium_risk_detected or intensity in [EmotionIntensity.SEVERE, EmotionIntensity.EXTREME]:
            return self._create_medium_risk_assessment(medium_triggers, intensity, emotion)
        
        # Step 6: Low risk - normal conversation
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
    
    def _assess_intensity(self, emotion: str, confidence: float, text: str) -> EmotionIntensity:
        """
        Assess emotional intensity based on emotion type, confidence, and text patterns
        """
        # Intensity indicators
        extreme_words = ['extremely', 'unbearably', 'completely', 'totally', 'absolutely']
        severe_words = ['very', 'really', 'so', 'too', 'incredibly']
        
        has_extreme = any(word in text for word in extreme_words)
        has_severe = any(word in text for word in severe_words)
        
        # High-intensity emotions
        intense_emotions = ['angry', 'fearful', 'sad']
        
        # Determine intensity
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