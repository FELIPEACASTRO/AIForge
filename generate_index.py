import re

def generate_index_content(input_file, output_file):
    """
    Reads the processed resource list, filters out unwanted entries, and generates
    the bilingual INDEX.md content with a Markdown table.
    """
    
    # Bilingual Header
    header = """# AIForge - Alphabetical Index / Índice Alfabético

## 🇬🇧 English

This index provides an alphabetical listing of all 15,686+ resources curated in the AIForge repository.

| Title | URL |
| :--- | :--- |
"""
    
    # Bilingual Separator
    separator = """
---

## 🇧🇷 Português

Este índice fornece uma listagem alfabética de todos os mais de 15.686 recursos curados no repositório AIForge.

| Título | URL |
| :--- | :--- |
"""

    # Final content structure
    content = header
    
    # Read and filter resources
    with open(input_file, 'r', encoding='utf-8') as f:
        resources = f.readlines()

    # Filter criteria:
    # 1. Must contain a '|' separator.
    # 2. Must not contain '!' (badges/images).
    # 3. Must not start with '[' (malformed links).
    # 4. Must not contain 'CONTRIBUTING.md' (internal link that slipped through).
    # 5. Must not contain 'LICENSE' (license badges).
    
    filtered_resources = []
    for line in resources:
        line = line.strip()
        if '|' in line and '!' not in line and not line.startswith('[') and 'CONTRIBUTING.md' not in line and 'LICENSE' not in line:
            # Re-format to ensure clean table row
            title, url = line.split('|', 1)
            title = title.strip()
            url = url.strip()
            
            # Re-create the Markdown link for the title
            markdown_link = f"[{title}]({url})"
            
            # Create the table row
            table_row = f"| {markdown_link} | {url} |"
            filtered_resources.append(table_row)

    # Sort the resources alphabetically by title (the first element in the table row)
    # The sort key is the text inside the first set of brackets [Title]
    def sort_key(row):
        match = re.search(r'\[(.*?)\]', row)
        return match.group(1).lower() if match else row.lower()

    filtered_resources.sort(key=sort_key)
    
    # Add resources to content
    content += "\n".join(filtered_resources)
    content += "\n"
    content += separator
    content += "\n".join(filtered_resources) # Re-use the same list for the Portuguese section
    content += "\n"

    # Write the final content to INDEX.md
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(content)

if __name__ == "__main__":
    input_file = "/home/ubuntu/AIForge/all_resources_processed.txt"
    output_file = "/home/ubuntu/AIForge/INDEX.md"
    generate_index_content(input_file, output_file)
    print(f"INDEX.md generated successfully at {output_file}")
