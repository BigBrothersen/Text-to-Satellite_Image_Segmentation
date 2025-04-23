from NLP import NLPProcessor
import sys
import json
import os
from typing import List, Dict

def clear_file(filepath: str):
    """Clear the contents of a file or create it if it doesn't exist"""
    open(filepath, 'w').close()

# To run, use the command:
# Make sure you are in the directory where the script is located
# and run the following command in the terminal:
# python batch_process.py 1.in 1.out

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
            # Convert query to lowercase
            query = line.strip().lower()
            if not query:  # Skip empty lines
                continue
            
            # Split query by 'and' to handle multiple labels (maintain lowercase)
            sub_queries = [q.strip() for q in query.split(' and ')]
            all_results = []
            
            # Process each sub-query
            for sub_query in sub_queries:
                results = processor.process_query(sub_query)
                if results[0].get('label'):  # Only add if valid result found
                    all_results.extend(results)
            
            # Get unique results with highest confidence for each label
            unique_results = {}
            # Sort all results by confidence first
            sorted_results = sorted(all_results, key=lambda x: x['confidence'], reverse=True)
            for result in sorted_results:
                label = result['label']
                # Only add if label not seen yet (first occurrence will be highest confidence)
                if label and label not in unique_results:
                    unique_results[label] = {
                        'label': label,
                        'color': result['color']
                    }
            
            # Convert to list maintaining query order
            final_results = list(unique_results.values())
            
            # Write results to output file
            if final_results:
                for result in final_results:
                    fout.write(json.dumps(result) + '\n')
            else:
                # Write empty result if no matches found
                fout.write(json.dumps({'label': None, 'color': None}) + '\n')

def main():
    if len(sys.argv) != 3:
        print("Usage: python batch_process.py 1.in 1.out")
        sys.exit(1)
    
    # Get the directory where the script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    model_path = os.path.join(script_dir, "data-nlp", "csv", "satellite_image_queries.csv")
    
    process_batch(input_file, output_file, model_path)

if __name__ == "__main__":
    main()