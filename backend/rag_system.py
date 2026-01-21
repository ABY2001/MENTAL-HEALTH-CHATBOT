"""
RAG (Retrieval-Augmented Generation) System
Retrieves relevant mental health information from a knowledge base
"""

import os
import pickle
from typing import List, Dict, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

class MentalHealthKnowledgeBase:
    """
    Mental health knowledge base with coping strategies, techniques, and resources
    """
    
    def __init__(self):
        self.documents = {
            # Depression support
            'depression_overview': {
                'content': """Depression is a common but serious mood disorder. It causes severe symptoms that affect how you feel, think, and handle daily activities. Symptoms must last at least two weeks for a diagnosis. Depression is treatable with medication, psychotherapy, or both.""",
                'category': 'depression',
                'keywords': ['depression', 'sad', 'hopeless', 'worthless']
            },
            'depression_coping': {
                'content': """Coping strategies for depression: 1) Establish a routine and stick to it. 2) Set small, achievable goals each day. 3) Exercise regularly, even a short walk helps. 4) Get adequate sleep on a regular schedule. 5) Eat healthy, balanced meals. 6) Stay connected with supportive people. 7) Avoid alcohol and drugs. 8) Practice mindfulness or meditation.""",
                'category': 'depression',
                'keywords': ['cope', 'manage', 'deal with', 'help']
            },
            
            # Anxiety support
            'anxiety_overview': {
                'content': """Anxiety disorders are the most common mental health concern. They involve excessive fear or anxiety. Types include generalized anxiety disorder, panic disorder, and social anxiety disorder. Anxiety is highly treatable through therapy, medication, or lifestyle changes.""",
                'category': 'anxiety',
                'keywords': ['anxiety', 'anxious', 'worried', 'panic']
            },
            'anxiety_grounding': {
                'content': """5-4-3-2-1 grounding technique for anxiety: Name 5 things you can see, 4 things you can touch, 3 things you can hear, 2 things you can smell, and 1 thing you can taste. This helps bring you back to the present moment and reduce anxiety.""",
                'category': 'anxiety',
                'keywords': ['panic attack', 'grounding', 'technique', 'calm down']
            },
            'anxiety_breathing': {
                'content': """Deep breathing for anxiety: Breathe in slowly through your nose for 4 counts, hold for 4 counts, breathe out through your mouth for 6 counts. Repeat 5-10 times. This activates your parasympathetic nervous system and promotes calmness.""",
                'category': 'anxiety',
                'keywords': ['breathing', 'calm', 'relax', 'technique']
            },
            
            # Stress management
            'stress_management': {
                'content': """Stress management techniques: 1) Practice time management and prioritize tasks. 2) Exercise regularly to release tension. 3) Practice relaxation techniques like meditation or yoga. 4) Maintain social connections. 5) Get enough sleep. 6) Eat a healthy diet. 7) Limit caffeine and alcohol. 8) Set boundaries and learn to say no.""",
                'category': 'stress',
                'keywords': ['stress', 'stressed', 'overwhelmed', 'pressure']
            },
            
            # Self-care
            'self_care': {
                'content': """Self-care basics: 1) Physical: regular exercise, healthy eating, adequate sleep, medical care. 2) Emotional: therapy, journaling, expressing feelings, setting boundaries. 3) Social: spending time with loved ones, joining support groups. 4) Spiritual: meditation, nature, values alignment. 5) Professional: work-life balance, pursuing interests.""",
                'category': 'self-care',
                'keywords': ['self-care', 'take care', 'wellbeing', 'health']
            },
            
            # Crisis resources
            'crisis_support': {
                'content': """If you're in crisis: 1) Call 988 (Suicide & Crisis Lifeline) - available 24/7. 2) Text HOME to 741741 (Crisis Text Line). 3) Call 1-800-662-4357 (SAMHSA National Helpline). 4) Go to your nearest emergency room. 5) Call 911 if in immediate danger. You are not alone, and help is available.""",
                'category': 'crisis',
                'keywords': ['crisis', 'emergency', 'suicide', 'help now']
            },
            
            # Therapy information
            'therapy_types': {
                'content': """Types of therapy: 1) Cognitive Behavioral Therapy (CBT): Focuses on changing negative thought patterns. 2) Dialectical Behavior Therapy (DBT): Teaches emotional regulation and distress tolerance. 3) Psychodynamic therapy: Explores past experiences. 4) Interpersonal therapy: Improves relationships. 5) Group therapy: Shared experiences with others.""",
                'category': 'therapy',
                'keywords': ['therapy', 'counseling', 'treatment', 'therapist']
            },
            
            # Medication information
            'medication_info': {
                'content': """Mental health medications: Antidepressants (SSRIs, SNRIs) treat depression and anxiety. Anti-anxiety medications provide short-term relief. Mood stabilizers help bipolar disorder. Antipsychotics treat severe symptoms. Always consult a psychiatrist for medication management. Medication works best combined with therapy.""",
                'category': 'medication',
                'keywords': ['medication', 'medicine', 'pills', 'psychiatrist']
            },
            
            # Sleep hygiene
            'sleep_hygiene': {
                'content': """Good sleep hygiene: 1) Keep a consistent sleep schedule. 2) Create a relaxing bedtime routine. 3) Make your bedroom cool, dark, and quiet. 4) Avoid screens 1 hour before bed. 5) Don't consume caffeine after 2pm. 6) Exercise regularly, but not right before bed. 7) Avoid large meals before bedtime.""",
                'category': 'sleep',
                'keywords': ['sleep', 'insomnia', 'cant sleep', 'tired']
            },
            
            # Mindfulness
            'mindfulness_practice': {
                'content': """Mindfulness practice: Focus on the present moment without judgment. Start with 5 minutes daily. Pay attention to your breathing, bodily sensations, thoughts, and feelings. When your mind wanders, gently bring it back. Benefits include reduced stress, improved focus, and better emotional regulation.""",
                'category': 'mindfulness',
                'keywords': ['mindfulness', 'meditation', 'present', 'awareness']
            },
            
            # Social connection
            'social_support': {
                'content': """Building social support: 1) Reach out to friends and family regularly. 2) Join clubs or groups with shared interests. 3) Volunteer in your community. 4) Attend support groups for your specific concerns. 5) Be vulnerable and share your feelings. 6) Practice active listening. 7) Set healthy boundaries in relationships.""",
                'category': 'social',
                'keywords': ['lonely', 'isolated', 'friends', 'connection']
            },
            
            # Anger management
            'anger_management': {
                'content': """Managing anger: 1) Recognize early warning signs (tension, rapid heartbeat). 2) Take a timeout and remove yourself from the situation. 3) Practice deep breathing or counting to 10. 4) Exercise to release tension. 5) Express feelings assertively, not aggressively. 6) Use "I" statements. 7) Practice forgiveness. 8) Seek professional help if needed.""",
                'category': 'anger',
                'keywords': ['angry', 'rage', 'furious', 'mad']
            },
            
            # Grief support
            'grief_support': {
                'content': """Coping with grief: 1) Allow yourself to feel all emotions without judgment. 2) Take care of your physical health. 3) Talk about your loss with supportive people. 4) Join a grief support group. 5) Maintain routines when possible. 6) Be patient with yourself - grief has no timeline. 7) Consider professional counseling. 8) Create meaningful rituals to honor your loss.""",
                'category': 'grief',
                'keywords': ['grief', 'loss', 'death', 'mourning']
            }
        }

class RAGSystem:
    """
    Retrieval-Augmented Generation System for Mental Health Support
    """
    
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        """
        Initialize RAG system with sentence transformer and FAISS index
        
        Args:
            model_name: SentenceTransformer model to use for embeddings
        """
        print("Initializing RAG System...")
        
        # Initialize knowledge base
        self.knowledge_base = MentalHealthKnowledgeBase()
        
        # Initialize embedding model
        print(f"Loading embedding model: {model_name}")
        self.embedding_model = SentenceTransformer(model_name)
        
        # Initialize FAISS index
        self.index = None
        self.document_ids = []
        
        # Build index
        self._build_index()
        
        print("✓ RAG System initialized successfully")
    
    def _build_index(self):
        """
        Build FAISS index from knowledge base documents
        """
        print("Building FAISS index from knowledge base...")
        
        # Extract all documents
        documents = []
        doc_ids = []
        
        for doc_id, doc_data in self.knowledge_base.documents.items():
            documents.append(doc_data['content'])
            doc_ids.append(doc_id)
        
        # Generate embeddings
        print(f"Generating embeddings for {len(documents)} documents...")
        embeddings = self.embedding_model.encode(documents, show_progress_bar=False)
        
        # Create FAISS index
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)  # L2 distance
        self.index.add(embeddings.astype('float32'))
        
        self.document_ids = doc_ids
        
        print(f"✓ Index built with {len(documents)} documents")
    
    def retrieve(self, query: str, top_k: int = 3) -> List[Dict]:
        """
        Retrieve most relevant documents for a query
        
        Args:
            query: User query or emotion context
            top_k: Number of documents to retrieve
        
        Returns:
            List of relevant documents with scores
        """
        # Generate query embedding
        query_embedding = self.embedding_model.encode([query])
        
        # Search in FAISS index
        distances, indices = self.index.search(
            query_embedding.astype('float32'), 
            top_k
        )
        
        # Retrieve documents
        results = []
        for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
            if idx < len(self.document_ids):
                doc_id = self.document_ids[idx]
                doc_data = self.knowledge_base.documents[doc_id]
                
                results.append({
                    'id': doc_id,
                    'content': doc_data['content'],
                    'category': doc_data['category'],
                    'score': float(distance),
                    'rank': i + 1
                })
        
        return results
    
    def get_context_for_emotion(self, emotion: str, user_text: str = "") -> str:
        """
        Get relevant context based on emotion and user text
        
        Args:
            emotion: Detected emotion (sad, angry, fearful, etc.)
            user_text: User's original message
        
        Returns:
            Concatenated relevant context from knowledge base
        """
        # Build search query
        query = f"{emotion} {user_text}"
        
        # Retrieve relevant documents
        results = self.retrieve(query, top_k=2)
        
        # Build context
        context_parts = []
        for result in results:
            context_parts.append(result['content'])
        
        return "\n\n".join(context_parts) if context_parts else ""
    
    def save_index(self, path: str = "rag_data"):
        """Save FAISS index and metadata"""
        os.makedirs(path, exist_ok=True)
        
        # Save FAISS index
        faiss.write_index(self.index, os.path.join(path, "faiss_index.bin"))
        
        # Save document IDs
        with open(os.path.join(path, "document_ids.pkl"), 'wb') as f:
            pickle.dump(self.document_ids, f)
        
        print(f"✓ Index saved to {path}")
    
    def load_index(self, path: str = "rag_data"):
        """Load FAISS index and metadata"""
        # Load FAISS index
        self.index = faiss.read_index(os.path.join(path, "faiss_index.bin"))
        
        # Load document IDs
        with open(os.path.join(path, "document_ids.pkl"), 'rb') as f:
            self.document_ids = pickle.load(f)
        
        print(f"✓ Index loaded from {path}")

# Example usage
if __name__ == "__main__":
    # Initialize RAG system
    rag = RAGSystem()
    
    # Test queries
    test_cases = [
        ("I'm feeling very anxious and can't calm down", "fearful"),
        ("I'm so depressed I can barely get out of bed", "sad"),
        ("I can't sleep at night", "neutral"),
        ("I'm furious and can't control my anger", "angry"),
    ]
    
    print("\n" + "="*70)
    print("RAG SYSTEM - TEST RETRIEVAL")
    print("="*70)
    
    for user_text, emotion in test_cases:
        print(f"\nQuery: '{user_text}' (Emotion: {emotion})")
        print("-"*70)
        
        context = rag.get_context_for_emotion(emotion, user_text)
        print(f"Retrieved Context:\n{context[:200]}...")
        print()
    
    # Save index
    rag.save_index()