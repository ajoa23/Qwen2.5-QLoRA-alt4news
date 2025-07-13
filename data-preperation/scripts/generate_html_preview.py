import json
import os
from pathlib import Path
from collections import defaultdict

INPUT_JSON = "results/sample_review.json"
OUTPUT_HTML = "results/sample_alt_text_preview.html"

def load_data(path):
    """Load the JSON data for visualization."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def generate_html(data):
    """Generate an HTML page to review Alt-Texts with navigation and previews."""
    html = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Alt-Text Review</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            background-color: #fafafa;
            margin: 0;
            display: flex;
        }
        nav {
            position: sticky;
            top: 0;
            align-self: flex-start;
            background: #ffffff;
            padding: 20px;
            min-width: 250px;
            height: 100vh;
            box-shadow: 2px 0 5px rgba(0,0,0,0.1);
            overflow-y: auto;
        }
        nav h2 { font-size: 1.2rem; margin-bottom: 10px; }
        nav ul { list-style: none; padding-left: 0; }
        nav li { margin-bottom: 8px; }
        nav a { text-decoration: none; color: #0077cc; }
        main { padding: 20px; flex-grow: 1; }
        .section { margin-bottom: 60px; }
        .section h2 { margin-top: 40px; border-bottom: 2px solid #ccc; padding-bottom: 5px; }
        .grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(350px, 1fr)); gap: 20px; }
        .card { 
            background: white; 
            border-radius: 8px; 
            box-shadow: 0 0 8px rgba(0,0,0,0.1); 
            padding: 15px; 
            display: flex; 
            flex-direction: column; 
            border: 3px solid transparent; 
        }
        .card.short { border-color: #0077cc; } /* Blue border */
        .card.long { border-color: #cc0000; } /* Red border */
        .card img { max-width: 100%; border-radius: 4px; }
        .card h3 { font-size: 1.1rem; margin: 10px 0 5px; }
        .card p { margin: 4px 0; font-size: 0.9rem; }
        .card a { text-decoration: none; color: #0077cc; font-weight: bold; }
        .alt-text { background: #f0f0f0; padding: 6px 8px; border-radius: 4px; margin-top: 6px; font-family: monospace; }
        .sample-type { font-size: 0.85rem; font-weight: bold; margin-top: 8px; }
    </style>
</head>
<body>
    <nav>
        <h2>Navigation</h2>
        <ul>
    """

    sections = sorted(set(entry.get("section", "Unknown Section") for entry in data))
    for section in sections:
        section_id = section.replace(" ", "-").lower()
        html += f'<li><a href="#{section_id}">{section}</a></li>'

    html += """
        </ul>
    </nav>
    <main>
        <h1>Alt-Text Review</h1>
    """

    section_entries = defaultdict(list)
    for entry in data:
        section = entry.get("section", "Unknown Section")
        section_entries[section].append(entry)

    for section in sections:
        section_id = section.replace(" ", "-").lower()
        html += f'<div class="section" id="{section_id}">'
        html += f'<h2>{section}</h2><div class="grid">'

        for entry in section_entries[section]:
            image = entry.get("image_url_clean", "")
            headline = entry.get("headline", "No Title")
            url = entry.get("article_url", "#")
            abstract = entry.get("abstract", "")
            caption = entry.get("caption", "")
            alt_initial = entry.get("openai_alt_text_initial", "—")
            alt_refined = entry.get("openai_alt_text_refined", "—")
            sample_type = entry.get("sample_type", "random").lower()

            # Determine CSS class for the border color
            sample_class = ""
            if sample_type == "short":
                sample_class = "short"
            elif sample_type == "long":
                sample_class = "long"

            html += f"""
            <div class="card {sample_class}">
                <img src="{image}" alt="Article Image">
                <h3><a href="{url}" target="_blank">{headline}</a></h3>
                <p><strong>Abstract:</strong> {abstract}</p>
                <p><strong>Caption:</strong> {caption}</p>
                <p><strong>Initial Alt-Text:</strong></p>
                <div class="alt-text">{alt_initial}</div>
                <p><strong>Refined Alt-Text:</strong></p>
                <div class="alt-text">{alt_refined}</div>
                <p class="sample-type">Sample Type: {sample_type.capitalize()}</p>
            </div>
            """

        html += "</div></div>"

    html += """
    </main>
</body>
</html>
    """
    return html

def main():
    print("[INFO] Loading JSON data...")
    data = load_data(INPUT_JSON)

    print(f"[INFO] Generating HTML preview for {len(data)} entries...")
    html_content = generate_html(data)

    Path(os.path.dirname(OUTPUT_HTML)).mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_HTML, "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"[INFO] HTML file saved to: {OUTPUT_HTML}")

if __name__ == "__main__":
    main()
