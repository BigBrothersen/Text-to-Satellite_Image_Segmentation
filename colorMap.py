import os
import json
from typing import Dict, Tuple

# Original color mapping
ORIGINAL_COLOUR_MAPPING = {
    (0, 255, 255): 0,    # Urban (Cyan)
    (255, 255, 0): 1,    # Agriculture (Yellow)
    (255, 0, 255): 2,    # Rangeland (Magenta)
    (0, 255, 0): 3,      # Forest (Green)
    (0, 0, 255): 4,      # Water (Blue)
    (255, 255, 255): 5,  # Barren (White)
    (0, 0, 0): 6         # Unknown (Black)
}

# Label indices to names mapping
LABELS_MAPPING = {
    0: 'urban',
    1: 'agriculture',
    2: 'rangeland',
    3: 'forest',
    4: 'water',
    5: 'barren',
    6: 'unknown'
}

# Color name to RGB mapping
COLOR_TO_RGB = {
    'cyan': (0, 255, 255),
    'yellow': (255, 255, 0),
    'magenta': (255, 0, 255),
    'green': (0, 255, 0),
    'blue': (0, 0, 255),
    'white': (255, 255, 255),
    'black': (0, 0, 0)
}

# Label to original color mapping
LABEL_TO_DEFAULT_COLOR = {
    'urban': 'cyan',
    'agriculture': 'yellow',
    'rangeland': 'magenta',
    'forest': 'green',
    'water': 'blue',
    'barren': 'white'
}

def process_color_mapping(nlp_output_file: str) -> Dict[Tuple[int, int, int], int]:
    """
    Process NLP output and create new color mapping.
    Maintains original indices but updates RGB values based on NLP output.
    """
    # Initialize all indices with black color (unknown)
    new_mapping = {(0, 0, 0): i for i in range(7)}
    
    # Read NLP output
    with open(nlp_output_file, 'r') as f:
        results = [json.loads(line) for line in f]
    
    # Process each result
    for result in results:
        label = result.get('label', '').lower()
        color = result.get('color')
        
        # Find the index for this label
        label_index = None
        for _, idx in ORIGINAL_COLOUR_MAPPING.items():
            if LABELS_MAPPING[idx].lower() == label:
                label_index = idx
                break
                
        if label_index is not None:
            if color and color in COLOR_TO_RGB:
                # Use specified color from NLP output
                rgb = COLOR_TO_RGB[color]
            else:
                # Use default color for this label if no color specified
                default_color = LABEL_TO_DEFAULT_COLOR.get(label)
                rgb = COLOR_TO_RGB[default_color]
            
            # Update mapping maintaining the original index
            new_mapping[rgb] = label_index
    
    return new_mapping

def save_mapping(mapping: Dict[Tuple[int, int, int], int], output_file: str):
    """
    Save color mapping to file in ordered format.
    Ensures all indices 0-6 are present and in order.
    """
    # Create reverse mapping (index to RGB)
    index_to_rgb = {idx: (0, 0, 0) for idx in range(7)}  # Default all to black
    for rgb, idx in mapping.items():
        index_to_rgb[idx] = rgb
    
    # Write in order with comments
    with open(output_file, 'w') as f:
        labels = ["Urban", "Agriculture", "Rangeland", "Forest", "Water", "Barren", "Unknown"]
        for i in range(7):
            rgb = index_to_rgb[i]
            f.write(f"{rgb}: {i}    # {labels[i]}\n")

def get_user_input() -> str:
    """Get queries from user input"""
    print("\nEnter your queries (one per line)")
    print("Enter an empty line to finish input")
    print("Example: show forests and urban areas")
    
    queries = []
    while True:
        query = input("> ").strip()
        if not query:
            break
        queries.append(query)
    
    return "\n".join(queries)

def save_input(text: str, input_file: str):
    """Save user input to file"""
    with open(input_file, 'w') as f:
        f.write(text)

def main():
    # Get the directory where the script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    nlp_dir = os.path.join(script_dir, "NLP")
    
    # Get user input and save to 1.in
    user_input = get_user_input()
    input_file = os.path.join(nlp_dir, "1.in")
    save_input(user_input, input_file)
    
    # Check if model exists, if not run NLP processing
    model_path = os.path.join(nlp_dir, "models", "bert_embeddings.pkl")
    if not os.path.exists(model_path):
        print("Model not found. Running NLP processing...")
        os.chdir(nlp_dir)
        os.system("python batch_process.py 1.in 1.out")
    else:
        print("Model found. Processing color mapping...")
    
    # Process NLP output and create color mapping
    nlp_output = os.path.join(nlp_dir, "1.out")
    print("Running batch processing...")
    os.chdir(nlp_dir)
    os.system("python batch_process.py 1.in 1.out")
    
    # Create new color mapping
    new_mapping = process_color_mapping(nlp_output)
    
    # Save results
    os.chdir(script_dir)
    save_mapping(new_mapping, "result.out")
    print("\nColor mapping saved to result.out")

if __name__ == "__main__":
    main()