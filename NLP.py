import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from typing import Dict, Union, List
import joblib
import os

class NLPProcessor:
    def __init__(self, csv_path: str, model_dir: str = "models"):
        """
        Initialize the NLP processor with training data using TF-IDF
        Args:
            csv_path: Path to training data CSV
            model_dir: Directory to save/load model files
        """
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)
        
        # Load or create new model
        model_path = os.path.join(model_dir, "tfidf_vectorizer.pkl")
        if os.path.exists(model_path):
            self.load_model(model_path)
        else:
            self.train_model(csv_path)
    
    def train_model(self, csv_path: str):
        """Train the model with new data"""
        self.df = pd.read_csv(csv_path)
        self.label_to_color = dict(zip(self.df['label'].unique(), 
                                     self.df['color'].unique()))
        
        # Initialize and fit TF-IDF
        self.vectorizer = TfidfVectorizer()
        self.query_embeddings = self.vectorizer.fit_transform(self.df['text'].tolist())
    
    def save_model(self, model_path: str = None):
        """Save the trained model and mappings"""
        if model_path is None:
            model_path = os.path.join(self.model_dir, "tfidf_vectorizer.pkl")
            
        # Save the model state
        model_state = {
            'vectorizer': self.vectorizer,
            'query_embeddings': self.query_embeddings,
            'label_to_color': self.label_to_color,
            'training_data': self.df
        }
        joblib.dump(model_state, model_path)
        print(f"Model saved to {model_path}")
    
    def load_model(self, model_path: str):
        """Load a trained model"""
        model_state = joblib.load(model_path)
        self.vectorizer = model_state['vectorizer']
        self.query_embeddings = model_state['query_embeddings']
        self.label_to_color = model_state['label_to_color']
        self.df = model_state['training_data']
        print(f"Model loaded from {model_path}")

    def process_query(self, query: str, threshold: float = 0.7) -> Dict[str, Union[str, float]]:
        """
        Process a natural language query and return structured output
        Args:
            query: Input text query
            threshold: Minimum similarity score to consider a match valid
        """
        try:
            # Encode input query
            query_embedding = self.vectorizer.transform([query])
            
            # Calculate similarities with all training queries
            similarities = cosine_similarity(query_embedding, self.query_embeddings)[0]
            
            # Find best match
            best_idx = np.argmax(similarities)
            confidence = float(similarities[best_idx])
            
            if confidence >= threshold:
                best_match = self.df.iloc[best_idx]
                return {
                    'label': best_match['label'],
                    'color': best_match['color'],
                    'confidence': confidence,
                    'matched_query': best_match['text']
                }
            else:
                return {
                    'label': None,
                    'color': None,
                    'confidence': confidence,
                    'error': 'No matching query found above threshold'
                }

        except Exception as e:
            print(f"Error processing query: {str(e)}")
            return {
                'label': None,
                'color': None,
                'confidence': 0.0,
                'error': str(e)
            }

    def get_valid_labels(self) -> Dict[str, str]:
        """Return the valid labels and their corresponding colors"""
        return self.label_to_color

    def batch_process(self, queries: List[str]) -> List[Dict[str, Union[str, float]]]:
        """Process multiple queries at once"""
        return [self.process_query(query) for query in queries]

# Example usage
if __name__ == "__main__":
    import time
    
    # Initialize the processor
    csv_path = "data-nlp/satellite_image_queries.csv"
    processor = NLPProcessor(csv_path)
    
    # Save the model after training
    processor.save_model()
    
    # Test queries
    test_queries = [
        "Please highlight the urban areas",
        "Show me the forests in green",
        "Color the water bodies in blue",
        "Display agricultural zones",
        "What about the desert regions?"
    ]

    # Process and display results with timing
    start_time = time.time()
    for query in test_queries:
        result = processor.process_query(query)
        print(f"\nQuery: {query}")
        print(f"Result: {result}")
    print(f"\nProcessing time: {time.time() - start_time:.2f} seconds")