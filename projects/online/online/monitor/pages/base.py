import base64
import logging
import os
from pathlib import Path

# where the pages are served from when running as the production
# user on the LDAS cluster
PUBLIC_HTML = Path("/home/aframe.online/public_html")
PUBLIC_URL = "https://ldas-jobs.ligo.caltech.edu/~aframe.online"


class MonitorPage:
    def __init__(
        self, run_dir: Path, out_dir: Path, logger: logging.Logger
    ) -> None:
        self.run_dir = run_dir
        self.out_dir = out_dir
        self.logger = logger

        self.source_event_dir = run_dir / "output" / "events"
        self.log_dir = run_dir / "output" / "logs"
        self.output_event_dir = out_dir / "events"
        if not self.output_event_dir.exists():
            self.output_event_dir.mkdir(exist_ok=True, parents=True)

        self.dataframe_file = out_dir / "event_data.hdf5"

    @property
    def root_url(self):
        """
        The root url for the output of the monitoring process, which
        is served from under PUBLIC_HTML.
        """
        return f"{PUBLIC_URL}/{self.out_dir.relative_to(PUBLIC_HTML)}"

    def write_atomic(self, path: Path, content: str) -> None:
        """
        Write a file so that a consumer never seems a partially written file.
        """
        tmp = path.with_suffix(path.suffix + ".tmp")
        with open(tmp, "w") as f:
            f.write(content)
        os.replace(tmp, path)

    @property
    def summary_page_url(self):
        return self.root_url + "/summary.html"

    @property
    def event_root_url(self):
        return self.root_url + "/events"

    def html_header(self, label: str) -> str:
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
                .table-container {{
                    background: white;
                    padding: 10px;
                    border-radius: 8px;
                    box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                    margin: 20px auto;
                    max-width: 820px;
                    overflow-x: auto;
                }}
                table {{
                    border-collapse: collapse;
                    width: 100%;
                }}
                th, td {{
                    border-bottom: 1px solid #ddd;
                    padding: 6px 10px;
                    text-align: left;
                    white-space: nowrap;
                }}
                th {{
                    color: #333;
                }}
            </style>
        </head>
        """
        return html_header

    def html_body(self) -> str:
        raise NotImplementedError

    def html_footer(self) -> str:
        html_footer = """
            </div>
        </body>
        </html>
        """
        return html_footer

    def html_table(
        self, headers: list[str], rows: list[list[str]], caption: str
    ) -> str:
        """
        Render a table.

        Args:
            headers: Column headers.
            rows: Row values formatted for display.
            caption: Caption displayed above the table.
        """
        header_html = "".join(f"<th>{header}</th>" for header in headers)
        rows_html = "".join(
            "<tr>" + "".join(f"<td>{value}</td>" for value in row) + "</tr>"
            for row in rows
        )
        return f"""
            <div class="table-container">
                <div class="caption">{caption}</div>
                <table>
                    <thead><tr>{header_html}</tr></thead>
                    <tbody>{rows_html}</tbody>
                </table>
            </div>
        """

    def embed_image(self, image_path: Path, caption: str) -> str:
        """
        Embed an image in HTML using base64 encoding.

        Args:
            image_path: Path to the image file.
            caption: Caption for the image.

        Returns:
            str: HTML string with the embedded image.
        """
        with open(image_path, "rb") as img_file:
            encoded = base64.b64encode(img_file.read()).decode("utf-8")
        return f'''
            <div class="item">
                <img src="data:image/png;base64,{encoded}" alt="{caption}">
                <div class="caption">{caption}</div>
            </div>
        '''
