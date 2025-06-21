import base64
from pathlib import Path


def html_header(label: str) -> str:
    """
    Generate the HTML header with a title.

    Args:
        label: Title for the HTML page.

    Returns:
        str: HTML header string.
    """
    html_header = f"""<!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>{label}</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                background-color: #f4f4f4;
                text-align: center;
                margin: 0;
                padding: 20px;
            }}
            .gallery {{
                display: flex;
                flex-wrap: wrap;
                justify-content: center;
                gap: 20px;
            }}
            .item {{
                background: white;
                padding: 10px;
                border-radius: 8px;
                box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                width: 100%;
                max-width: 400px;
                box-sizing: border-box;
            }}
            img {{
                width: 100%;
                height: auto;
                border-radius: 5px;
            }}
            .caption {{
                margin-top: 8px;
                font-weight: bold;
                color: #333;
            }}
        </style>
    </head>
    """
    return html_header


def html_footer():
    html_footer = """
        </div>
    </body>
    </html>
    """
    return html_footer


def embed_image(image_path: Path, caption: str) -> str:
    """
    Embed an image in HTML using base64 encoding.

    Args:
        image_path: Path to the image file.
        caption: Caption for the image.

    Returns:
        str: HTML string with the embedded image.
    """
    with open(image_path, "rb") as img_file:
        encoded_string = base64.b64encode(img_file.read()).decode("utf-8")
    return f'''
        <div class="item">
            <img src="data:image/png;base64,{encoded_string}" alt="{caption}">
            <div class="caption">{caption}</div>
        </div>
    '''
