import pandas as pd
import numpy as np
from typing import Dict, Union, List
import joblib
import os
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

class NLPProcessor:
    def __init__(self, csv_path: str, model_dir: str = None, force_retrain: bool = False):
        """
        Initialize the NLP processor with BERT
        Args:
            csv_path: Path to training data CSV
            model_dir: Directory to save/load model files
            force_retrain: If True, retrain model even if it exists
        """
        # Get the directory where the script is located
        script_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Set model directory relative to script location
        if model_dir is None:
            self.model_dir = os.path.join(script_dir, "models")
        else:
            self.model_dir = model_dir
            
        os.makedirs(self.model_dir, exist_ok=True)
        
        # Initialize BERT model
        self.bert_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Load or create new model
        model_path = os.path.join(self.model_dir, "bert_embeddings.pkl")
        
        if os.path.exists(model_path) and not force_retrain:
            self.load_model(model_path)
        else:
            if csv_path is None:
                raise ValueError("CSV path required for training new model")
            print("Training new model...")
            self.train_model(csv_path)
            self.save_model(model_path)
    
    def train_model(self, csv_path: str):
        """Train the model with new data"""
        print("Loading training data...")
        self.df = pd.read_csv(csv_path)
        self.label_to_color = dict(zip(self.df['label'].unique(), 
                                     self.df['color'].unique()))
        
        # Generate BERT embeddings with progress bar
        print("Generating BERT embeddings...")
        texts = self.df['text'].tolist()
        self.query_embeddings = []
        
        for text in tqdm(texts, desc="Encoding texts"):
            embedding = self.bert_model.encode([text])[0]
            self.query_embeddings.append(embedding)
        self.query_embeddings = np.array(self.query_embeddings)
        print("Training complete!")
    
    def save_model(self, model_path: str = None):
        """Save the trained model and mappings"""
        if model_path is None:
            model_path = os.path.join(self.model_dir, "bert_embeddings.pkl")
            
        # Save the model state
        model_state = {
            'query_embeddings': self.query_embeddings,
            'label_to_color': self.label_to_color,
            'training_data': self.df
        }
        joblib.dump(model_state, model_path)
        print(f"Model saved to {model_path}")
    
    def load_model(self, model_path: str):
        """Load a trained model"""
        model_state = joblib.load(model_path)
        self.query_embeddings = model_state['query_embeddings']
        self.label_to_color = model_state['label_to_color']
        self.df = model_state['training_data']
        print(f"Model loaded from {model_path}")

    def process_query(self, query: str, initial_threshold: float = 0.90) -> List[Dict[str, Union[str, float]]]:
        """
        Process a natural language query and return structured output
        Args:
            query: Input text query
            initial_threshold: Initial threshold value (will decrease if no matches found)
        """
        try:
            query = query.lower()
            
            # More granular threshold progression
            thresholds = [0.90, 0.85, 0.80, 0.75, 0.70]
            
            for current_threshold in thresholds:
                # Encode input query using BERT
                query_embedding = self.bert_model.encode([query])[0]
                
                # Calculate similarities
                similarities = cosine_similarity([query_embedding], self.query_embeddings)[0]
                
                # Find matches above current threshold
                matches = []
                for idx, confidence in enumerate(similarities):
                    if confidence >= current_threshold:
                        match = self.df.iloc[idx]
                        color = None if pd.isna(match['color']) else match['color']
                        matches.append({
                            'label': match['label'],
                            'color': color,
                            'confidence': float(confidence),
                            'matched_query': match['text']
                        })
                
                # If matches found, return them
                if matches:
                    return sorted(matches, key=lambda x: x['confidence'], reverse=True)
            
            # If no matches found at any threshold
            return [{
                'label': None,
                'color': None,
                'confidence': 0.0,
                'error': 'No matching query found above minimum threshold'
            }]

        except Exception as e:
            print(f"Error processing query: {str(e)}")
            return [{
                'label': None,
                'color': None,
                'confidence': 0.0,
                'error': str(e)
            }]

    def get_valid_labels(self) -> Dict[str, str]:
        """Return the valid labels and their corresponding colors"""
        return self.label_to_color

    def batch_process(self, queries: List[str]) -> List[List[Dict[str, Union[str, float]]]]:
        """Process multiple queries at once"""
        results = []
        for query in tqdm(queries, desc="Processing queries"):
            results.append(self.process_query(query))
        return results

# Example usage
if __name__ == "__main__":
    import time
    import os
    
    print("Initializing NLP Processor...")
    start_time = time.time()
    
    # Get the directory where the script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Build the correct path to the CSV file
    csv_name = "satellite_image_queries.csv"  # Make sure this matches your actual filename
    csv_path = os.path.join(script_dir, "data-nlp", "csv", csv_name)
    
    # Verify file exists before proceeding
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found at: {csv_path}\nPlease check the filename and path.")
    
    print(f"Looking for CSV file at: {csv_path}")
    processor = NLPProcessor(csv_path, force_retrain=True)
    
    print(f"\nInitialization time: {time.time() - start_time:.2f} seconds")
    
    # Save the model after training
    print("\nSaving model...")
    processor.save_model()
    
    # Test queries
    test_queries = [
        "Show me forests and urban areas",
        "Highlight water bodies and agricultural zones",
        "Display desert regions and forests"
    ]

    print("\nTesting queries...")
    query_start_time = time.time()
    results = processor.batch_process(test_queries)
    
    # Display results
    for query, result in zip(test_queries, results):
        print(f"\nQuery: {query}")
        print(f"Results: {result}")
    
    print(f"\nQuery processing time: {time.time() - query_start_time:.2f} seconds")
    print(f"Total runtime: {time.time() - start_time:.2f} seconds")