import csv
import os
from itertools import product

# Define your variations
objects = {
    "urban": ["urban", "urbans", "urban areas", "cities", "city areas", "urban regions"],
    "agriculture": ["agriculture", "agricultures", "farm", "agricultural areas", "farmland", "farming areas", "crops"],
    "rangeland": ["rangeland", "grassland", "pastures", "grazing areas", "meadows"],
    "forest": ["forest", "forests", "forested areas", "woodland", "forest regions"],
    "water": ["water", "water bodies", "lakes", "rivers", "water areas"],
    "barren": ["barren", "desert", "arid land", "barren areas", "wasteland"]
}

colors = [
    "cyan", "yellow", "magenta", "green", "blue", "white", "None",
    "red", "light_blue", "light_green", "orange", "purple", "pink",
    "brown", "gray", "grey", "violet", "silver", "gold", "teal"
]

# Define text templates
templates = [
    "<object> <color>",
    "<object> in <color>",
    "<object> using <color>",
    "<object> with <color>",
    "<object> colored in <color>",
    "<object> color <color>",
    "<object> <color> color",
    "<object> <color> colour",
    "Colour <object> with <color>",
    ",<object> <color>",
    "and also <object> <color>",
    "also do <object> <color>",
    "and colour <object> <color>",
    "and <object> <color>",
    "Let <object> <color>",
    "Hi <object> <color>",
    "Hello <object> <color>",
    "Hey <object> <color>",
    "Go <color> <object>",
    "Make <object> <color>",
    "Set the <object> <color>",
    "<object> be <color>",
    "<object> is <color>",
    "<object> <color> please",
    "<object> <color> please!",
    "<object> <color> thx",
    "<object> <color> thanks",
    "<object> <color> please :)",
    "I want <object> <color>",
    "Is the <object> <color>",
    "Can you show me the <object> <color>",
    "Please highlight the <object> <color>",
    "Please do <object> <color>",
    ",and <object> <color>",
    "Show me the <object> <color>",
    "The <object> <color>",
    "The color of <object> is <color>",
    "The <object> is <color>",
    "Do <object> <color>",
    "Also <object> <color>",
    "Want to <object> <color>",
    "Color <object> <color>",
    "Show <object> <color>",
    "Display <object> <color>",
    "Present <object> <color>",
    "Reveal <object> <color>",
    "Exhibit <object> <color>",
    "Demonstrate <object> <color>",
    "Illustrate <object> <color>",
    "Expose <object> <color>",
    "Unveil <object> <color>",
    "Present me the <object> <color>",
    "Let me see the <object> <color>",
    "Identify the <object> <color>",
    "Indicate the <object> <color>",
    "Look for the <object> <color>",
    "Find the <object> <color>",
    "Locate the <object> <color>",
    "Point out the <object> <color>",
    "Could you display the <object> <color>",
    "Please show the <object> <color>",
    "Hello, I want <object> to be the color of <color>",
    "Can you highlight the <object> using <color>",
    "Make the <object> appear in <color>",
    "Do the <object> appear in <color>",
    "I would like to see the <object> colored in <color>",
    "Display <object> <color>",
    "Mark all <object> with <color>"
]

def generate_queries():
    """Generate all possible combinations of queries"""
    rows = [["text", "label", "color"]]  # CSV header
    
    # Generate all combinations of templates, objects, and colors
    for template in templates:
        for label, variations in objects.items():
            for obj_variation in variations:
                for color in colors:
                    if color == "None":
                        # Remove color placeholder and any surrounding spaces/words
                        text = template
                        text = text.replace(" <color>", "")  # Remove color and space before it
                        text = text.replace("<color> ", "")  # Remove color and space after it
                        text = text.replace("<color>", "")   # Remove any remaining color placeholder
                        text = text.replace("in", "")       # Remove prepositions related to color
                        text = text.replace("using", "")
                        text = text.replace("with", "")
                        text = text.replace("colored in", "")
                        text = text.replace("appear in", "")
                        text = text.replace("is", "")
                        # Replace object placeholder
                        text = text.replace("<object>", obj_variation)
                        # Clean up any double spaces
                        text = " ".join(text.split())
                        rows.append([text, label, "None"])  # Keep the original label
                    else:
                        # Normal case - replace both placeholders
                        text = template.replace("<object>", obj_variation)
                        text = text.replace("<color>", color)
                        rows.append([text, label, color])
    
    return rows

def save_csv(rows, filename="satellite_image_queries.csv"):
    """Save queries to CSV file"""
    # Ensure the csv directory exists
    csv_dir = os.path.join(os.path.dirname(__file__), "csv")
    os.makedirs(csv_dir, exist_ok=True)
    
    # Full path to the CSV file
    filepath = os.path.join(csv_dir, filename)
    
    # Write to CSV file
    with open(filepath, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerows(rows)
    
    total_queries = len(rows) - 1  # Subtract 1 for header
    print(f"Generated {total_queries} queries")
    print(f"Templates used: {len(templates)}")
    print(f"Objects: {len(objects)} categories")
    print(f"Colors: {len(colors)} options")
    print(f"Saved to: {filepath}")

if __name__ == "__main__":
    # Generate and save queries
    queries = generate_queries()
    save_csv(queries)
    
    print("\nQueries generated and saved successfully.")
    # # Print some example queries
    # print("\nExample queries:")
    # for i, row in enumerate(queries[1:6], 1):  # Skip header, show first 5
    #     print(f"{i}. {row[0]}")
    #     print(f"   Label: {row[1]}, Color: {row[2]}")