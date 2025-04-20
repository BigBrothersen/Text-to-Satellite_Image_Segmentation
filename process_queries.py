from NLP import NLPProcessor
import datetime
import json
import os

class QueryProcessor:
    def __init__(self, model_dir: str = "models"):
        """Initialize the processor using saved model"""
        self.nlp = NLPProcessor(None, model_dir=model_dir)  # Pass None for csv_paths
        self.input_history = "data-nlp/inputHistory.txt"
        self.output_history = "data-nlp/outputHistory.txt"
        
        # Create directories if they don't exist
        os.makedirs("data-nlp", exist_ok=True)

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
                # Get user input
                query = input("> ").strip()
                
                # Check for exit condition
                if query.lower() == 'quit':
                    break
                
                # Generate unique ID
                query_id = self.generate_id()
                
                # Save input
                self.save_input(query, query_id)
                
                # Process query
                result = self.nlp.process_query(query)
                
                # Save output
                self.save_output(result, query_id)
                
                # Display result
                print("\nResult:")
                print(f"Query ID: {query_id}")
                print(json.dumps(result, indent=2))
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