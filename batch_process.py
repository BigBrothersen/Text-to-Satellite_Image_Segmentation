from NLP import NLPProcessor
import sys
import json
import os

def clear_file(filepath: str):
    """Clear the contents of a file or create it if it doesn't exist"""
    open(filepath, 'w').close()

# To run, use the command:
# python batch_process.py input.in output.out

def process_batch(input_file: str, output_file: str, model_path: str):
    """
    Process queries from input file and write results to output file
    Args:
        input_file: File containing one query per line
        output_file: File to write results to
        model_path: Path to the CSV file for NLP model
    """
    # Clear output file before processing
    clear_file(output_file)
    
    # Initialize NLP processor
    processor = NLPProcessor(model_path)
    
    # Process each query
    with open(input_file, 'r', encoding='utf-8') as fin, \
         open(output_file, 'w', encoding='utf-8') as fout:
        
        for line in fin:
            query = line.strip()
            if not query:  # Skip empty lines
                continue
                
            # Process query
            result = processor.process_query(query)
            
            # Write only label and color to output
            output = {
                'label': result['label'],
                'color': result['color']
            }
            fout.write(json.dumps(output) + '\n')

def main():
    if len(sys.argv) != 3:
        print("Usage: python batch_process.py input.in output.out")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    model_path = "data-nlp/satellite_image_queries.csv"
    
    process_batch(input_file, output_file, model_path)

if __name__ == "__main__":
    main()