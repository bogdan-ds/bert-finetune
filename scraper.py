import re
import requests
import uuid
import logging

from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


class WebScraper:

    def __init__(self,
                 save_locally: bool = False,
                 file_dir: str = "dataset/scraped-raw"):
        self.save_locally = save_locally
        self.file_dir = file_dir

    def get(self, url: str, timeout: int = 10) -> requests.Response:
        try:
            response = requests.get(url,
                                    timeout=timeout)
            if response.status_code == 200:
                return response
            else:
                logger.info(
                    f"Unexpected status code when making "
                    f"GET request to {url}: {response.status_code}")

        except Exception as e:
            logger.info(f"Could not make GET request to {url}, exception {e}")

    def extract_text_from_response(self, response: requests.Response) -> str:
        soup = BeautifulSoup(response.content, 'html.parser')

        # Remove all script and style elements
        for script in soup(["script", "style"]):
            script.extract()

        # Get the text content of the page
        text = soup.get_text()

        # Remove any extra white spaces and split by lines
        lines = (line.strip() for line in text.splitlines())

        # Break multi-headlines into separate lines and remove any empty lines
        chunks = (phrase.strip()
                  for line in lines for phrase in line.split("  "))
        text = ' '.join(chunk for chunk in chunks if chunk)

        # Generate a random filename for the text file and save
        if self.save_locally:
            sanitized_url = self.sanitize_url(response.url)
            filename = f"{self.file_dir}/{sanitized_url}.txt"
            with open(filename, 'w') as f:
                f.write(text)

        return text

    def sanitize_url(self, url: str) -> str:
        illegal_chars = r'[\\/:*?"<>|]'

        # Replace any illegal character with an underscore
        sanitized = re.sub(illegal_chars, '_', url)

        return sanitized
