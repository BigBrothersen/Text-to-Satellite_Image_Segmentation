from NLP import NLPProcessor
import datetime
import json
import os

class QueryProcessor:
    def __init__(self, model_dir: str = "models"):
        """Initialize the processor using saved model"""
        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.model_dir = os.path.join(script_dir, model_dir)
        self.input_history = os.path.join(script_dir, "data-nlp", "logs", "inputHistory.txt")
        self.output_history = os.path.join(script_dir, "data-nlp", "logs", "outputHistory.txt")
        
        # Create directories if they don't exist
        os.makedirs(os.path.join(script_dir, "data-nlp", "logs"), exist_ok=True)
        
        # Initialize NLP processor with correct model path
        self.nlp = NLPProcessor(None, model_dir=self.model_dir)

    def generate_id(self) -> str:
        """Generate a unique ID based on timestamp"""
        return datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")

    def save_input(self, query: str, query_id: str):
        """Save input query to history"""
        with open(self.input_history, "a", encoding="utf-8") as f:
            entry = {
                "id": query_id,
                "timestamp": datetime.datetime.now().isoformat(),
                "query": query
            }
            f.write(json.dumps(entry) + "\n")

    def save_output(self, result: dict, query_id: str):
        """Save output to history"""
        with open(self.output_history, "a", encoding="utf-8") as f:
            entry = {
                "id": query_id,
                "timestamp": datetime.datetime.now().isoformat(),
                "result": result
            }
            f.write(json.dumps(entry) + "\n")

    def process_input(self):
        """Process user input and save results"""
        print("\nEnter 'quit' to exit")
        print("Enter your query:")
        
        while True:
            try:
                # Get user input and convert to lowercase
                query = input("> ").strip().lower()
                
                # Check for exit condition
                if query == 'quit':
                    break
                
                # Generate unique ID
                query_id = self.generate_id()
                
                # Save input
                self.save_input(query, query_id)
                
                # Split query by 'and' to handle multiple labels
                sub_queries = [q.strip().lower() for q in query.split(' and ')]
                all_results = []
                
                # Process each sub-query with multiple thresholds
                for sub_query in sub_queries:
                    # Try different thresholds to find result with color
                    thresholds = [0.90, 0.85, 0.80]
                    best_result = None
                    
                    for threshold in thresholds:
                        results = self.nlp.process_query(sub_query, initial_threshold=threshold)
                        if results and results[0].get('label'):
                            # If we find a result with color, use it
                            if results[0].get('color') is not None:
                                best_result = results[0]
                                break
                            # Otherwise, keep the highest confidence result
                            elif best_result is None or results[0]['confidence'] > best_result['confidence']:
                                best_result = results[0]
                    
                    if best_result:
                        all_results.append(best_result)
                
                # If no results found, return error
                if not all_results:
                    all_results = [{
                        'label': None,
                        'color': None,
                        'confidence': 0.0,
                        'error': 'No matching labels found'
                    }]
                
                # Save output
                self.save_output(all_results, query_id)
                
                # Display result
                print("\nResult:")
                print(f"Query ID: {query_id}")
                print(f"Number of matches found: {len(all_results)}")
                print(json.dumps(all_results, indent=2))
                print("\nEnter next query:")
                
            except Exception as e:
                print(f"Error: {str(e)}")
                continue

def main():
    # Just use the saved model from models directory
    processor = QueryProcessor("models")  # We only need the model directory
    
    print("Natural Language Query Processor")
    print("================================")
    processor.process_input()

if __name__ == "__main__": 
    main()