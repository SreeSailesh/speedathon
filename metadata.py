import os
import json
import re
from datetime import datetime

import PyPDF2
import pdfminer.high_level
from pdfminer.high_level import extract_pages
from pdfminer.layout import LAParams, LTTextContainer, LTTextLine, LTChar

from PIL import Image
import exifread
import cv2
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# For DOCX processing
from docx import Document

# For DOC processing (legacy Word files)
import textract

# For audio metadata extraction
from mutagen import File as MutagenFile

# For HTML processing
from bs4 import BeautifulSoup

# Download necessary NLTK data (run once)
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("averaged_perceptron_tagger")
try:
    nltk.data.find("taggers/averaged_perceptron_tagger_eng")
except LookupError:
    nltk.download("averaged_perceptron_tagger_eng")


def remove_unicode(text):
    text_v = re.sub(r"[^\x00-\x7F]+", "", text)
    text_v = text_v.replace("\n", " ").replace("\t", " ")
    return re.sub(
        r"[^\x00-\x7F]+", "", text_v.replace("\u0000", "").replace("\u2022  ", "")
    )  # Removes all non-ASCII characters


def get_file_basic_metadata(file_path):
    """Return basic file metadata like file size and last modified date."""
    stats = os.stat(file_path)
    return {
        "file_size_bytes": stats.st_size,
        "last_modified": datetime.fromtimestamp(stats.st_mtime).isoformat(),
    }


def extract_plaintext_semantic_tags(text):
    """
    Extract semantic tags from plain text.
    - Frequency-based tags: tokenize, filter out stopwords, and keep only nouns.
    - Header detection: any line that is fully uppercase (and longer than 3 characters).
    Returns a dictionary with keys 'frequency_tags' and 'headers'.
    """
    global gl_text_title
    gl_text_title = ""
    semantic = {}
    # Frequency-based tags using only nouns.
    tokens = word_tokenize(remove_unicode(text))
    tokens = [word.lower() for word in tokens if word.isalpha()]
    filtered_words = [word for word in tokens if word not in stopwords.words("english")]
    tagged_tokens = nltk.pos_tag(filtered_words)
    content_words = [word for word, pos in tagged_tokens if pos.startswith("NN")]
    freq = {}
    for word in content_words:
        freq[word] = freq.get(word, 0) + 1
    sorted_words = sorted(freq.items(), key=lambda x: x[1], reverse=True)
    semantic["frequency_tags"] = [word for word, count in sorted_words[:10]]

    # Header detection: scan each line and flag if fully uppercase.
    headers = set()
    for line in text.splitlines():
        line = line.strip()
        if len(line) >= 3 and line.isupper():
            headers.add(line)
            if gl_text_title == "":
                gl_text_title = line
    semantic["headers"] = list(headers)
    return semantic


def extract_pdf_semantic_tags(file_path, clean_text):
    """
    Extract semantic tags from a PDF using multiple heuristics:
      - Frequency-based top words (using only nouns from cleaned text).
      - Bold text extraction.
      - Lines with a larger-than-normal average font size.
      - Header detection based on layout: fully uppercase or top-of-page.
    Returns a dictionary with keys 'frequency_tags', 'bold_texts', 'large_texts', and 'headers'.

    """
    global gl_pdf_title
    gl_pdf_title = ""
    semantic = {}
    # Frequency-based tags
    tokens = word_tokenize(remove_unicode(clean_text))
    tokens = [word.lower() for word in tokens if word.isalpha()]
    filtered_words = [word for word in tokens if word not in stopwords.words("english")]
    tagged_tokens = nltk.pos_tag(filtered_words)
    content_words = [word for word, pos in tagged_tokens if pos.startswith("NN")]
    freq = {}
    for word in content_words:
        freq[word] = freq.get(word, 0) + 1
    sorted_words = sorted(freq.items(), key=lambda x: x[1], reverse=True)
    semantic["frequency_tags"] = [word for word, count in sorted_words[:10]]

    # Layout-based heuristics: bold texts, large texts, and headers
    bold_texts = set()
    large_texts = set()
    headers = set()
    try:
        laparams = LAParams()
        for page_layout in extract_pages(file_path, laparams=laparams):
            page_height = page_layout.height
            for element in page_layout:
                if isinstance(element, LTTextContainer):
                    for text_line in element:
                        if isinstance(text_line, LTTextLine):
                            line_text = text_line.get_text().strip()
                            if not line_text:
                                continue
                            is_bold = False
                            font_sizes = []
                            for char in text_line:
                                if isinstance(char, LTChar):
                                    font_sizes.append(char.size)
                                    if "bold" in char.fontname.lower():
                                        is_bold = True
                            if is_bold:
                                bold_texts.add(line_text)
                            if font_sizes:
                                avg_font = sum(font_sizes) / len(font_sizes)
                                if avg_font >= 15:
                                    large_texts.add(line_text)
                                    # print(line_text)
                                    # print(gl_pdf_title)
                                    if gl_pdf_title == "":
                                        gl_pdf_title = line_text
                            # Identify headers: fully uppercase or positioned at the top 10% of the page.
                            if line_text.isupper() or (
                                text_line.bbox[3] > 0.9 * page_height
                            ):
                                headers.add(line_text)
    except Exception as e:
        semantic["layout_error"] = str(e)

    semantic["bold_texts"] = list(bold_texts)
    semantic["large_texts"] = list(large_texts)
    semantic["headers"] = list(headers)
    return semantic


def extract_pdf_metadata(file_path):
    metadata = get_file_basic_metadata(file_path)
    try:
        # Extract structural metadata using PyPDF2.
        with open(file_path, "rb") as f:
            pdf_reader = PyPDF2.PdfReader(f)
            metadata["num_pages"] = len(pdf_reader.pages)
            doc_info = pdf_reader.metadata
            if doc_info:
                metadata["title"] = doc_info.title
                metadata["author"] = doc_info.author

        # Extract text using pdfminer.six.
        raw_text = pdfminer.high_level.extract_text(file_path)
        clean_text = remove_unicode(
            re.sub(r"\s+", " ", remove_unicode(raw_text)).strip()
        )
        metadata["extracted_text"] = (
            clean_text[:500] + "..." if len(clean_text) > 500 else clean_text
        )

        # Semantic tags extraction.
        semantic_tags = extract_pdf_semantic_tags(file_path, clean_text)
        metadata["semantic_tags"] = semantic_tags

        # If title is missing, use first large text as title.
        if not metadata.get("title") and semantic_tags.get("large_texts"):
            large_texts = semantic_tags["large_texts"]
            if large_texts:
                metadata["title"] = gl_pdf_title
    except Exception as e:
        metadata["error"] = str(e)
    return metadata


def extract_image_metadata(file_path):
    metadata = get_file_basic_metadata(file_path)
    try:
        with Image.open(file_path) as img:
            metadata["format"] = img.format
            metadata["dimensions"] = {"width": img.width, "height": img.height}
            metadata["mode"] = img.mode

        # Extract EXIF metadata if available.
        with open(file_path, "rb") as f:
            tags = exifread.process_file(f, details=False)
            metadata["exif"] = {
                tag: str(value)
                for tag, value in tags.items()
                if tag
                not in ["JPEGThumbnail", "TIFFThumbnail", "Filename", "EXIF MakerNote"]
            }
    except Exception as e:
        metadata["error"] = str(e)
    return metadata


def extract_video_metadata(file_path):
    metadata = get_file_basic_metadata(file_path)
    try:
        cap = cv2.VideoCapture(file_path)
        if not cap.isOpened():
            metadata["error"] = "Cannot open video file"
            return metadata
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        metadata["frame_count"] = frame_count
        metadata["fps"] = fps
        metadata["dimensions"] = {"width": width, "height": height}
        if fps > 0:
            metadata["duration_seconds"] = frame_count / fps
        cap.release()
    except Exception as e:
        metadata["error"] = str(e)
    return metadata


def extract_audio_metadata(file_path):
    """Extract audio metadata using Mutagen."""
    metadata = get_file_basic_metadata(file_path)
    try:
        audio = MutagenFile(file_path)
        if audio is None:
            metadata["error"] = "Unsupported audio format or no metadata available."
            return metadata
        # Extract common tags if available.
        for tag in ["title", "artist", "album"]:
            if tag in audio.tags:
                metadata[tag] = str(audio.tags[tag])
        if audio.info and hasattr(audio.info, "length"):
            metadata["duration_seconds"] = audio.info.length
        metadata["audio_info"] = {
            k: str(v) for k, v in audio.info.__dict__.items() if not k.startswith("_")
        }
    except Exception as e:
        metadata["error"] = str(e)
    return metadata


def extract_txt_metadata(file_path):
    """Process plain text (.txt) files."""

    metadata = get_file_basic_metadata(file_path)
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            text = remove_unicode(f.read())
        metadata["extracted_text"] = text[:500] + "..." if len(text) > 500 else text
        metadata["semantic_tags"] = extract_plaintext_semantic_tags(text)
        if metadata["semantic_tags"].get("headers"):
            metadata["title"] = gl_text_title
    except Exception as e:
        metadata["error"] = str(e)
    return metadata


def extract_docx_metadata(file_path):
    """Process DOCX files using python-docx."""
    metadata = get_file_basic_metadata(file_path)
    try:
        doc = Document(file_path)
        full_text = remove_unicode("\n".join([para.text for para in doc.paragraphs]))
        metadata["extracted_text"] = (
            full_text[:500] + "..." if len(full_text) > 500 else full_text
        )
        metadata["semantic_tags"] = extract_plaintext_semantic_tags(full_text)
        if not metadata.get("title") and metadata["semantic_tags"].get("headers"):
            metadata["title"] = gl_text_title
    except Exception as e:
        metadata["error"] = str(e)
    return metadata


def extract_doc_metadata(file_path):
    """Process DOC files using textract."""
    metadata = get_file_basic_metadata(file_path)
    try:
        raw_text = textract.process(file_path)
        text = remove_unicode(raw_text.decode("utf-8", errors="ignore"))
        metadata["extracted_text"] = text[:500] + "..." if len(text) > 500 else text
        metadata["semantic_tags"] = extract_plaintext_semantic_tags(text)
        if not metadata.get("title") and metadata["semantic_tags"].get("headers"):
            metadata["title"] = gl_text_title
    except Exception as e:
        metadata["error"] = str(e)
    return metadata


def extract_html_metadata(file_path):
    """Process HTML files for metadata extraction."""
    metadata = get_file_basic_metadata(file_path)
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            html_content = f.read()
        soup = BeautifulSoup(html_content, "html.parser")
        # Extract title from <title> tag, if available.
        title_tag = soup.find("title")
        if title_tag and title_tag.string:
            metadata["title"] = title_tag.string.strip()
        # Extract text from the body.
        body = soup.find("body")
        if body:
            text = remove_unicode(body.get_text(separator="\n", strip=True))
        else:
            text = remove_unicode(soup.get_text(separator="\n", strip=True))
        metadata["extracted_text"] = text[:500] + "..." if len(text) > 500 else text
        metadata["semantic_tags"] = extract_plaintext_semantic_tags(text)
        # Use the first header from semantic tags if title is missing.
        if not metadata.get("title") and metadata["semantic_tags"].get("headers"):
            metadata["title"] = gl_text_title
    except Exception as e:
        metadata["error"] = str(e)
    return metadata


def process_files_in_folder(folder_path):
    results = {}
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path):
            ext = os.path.splitext(filename)[1].lower()
            if ext == ".pdf":
                meta = extract_pdf_metadata(file_path)
                results[filename] = {"type": "pdf", "metadata": meta}
            elif ext in [".jpg", ".jpeg", ".png", ".tiff", ".bmp"]:
                meta = extract_image_metadata(file_path)
                results[filename] = {"type": "image", "metadata": meta}
            elif ext in [".mp4", ".avi", ".mov", ".mkv"]:
                meta = extract_video_metadata(file_path)
                results[filename] = {"type": "video", "metadata": meta}
            elif ext in [".mp3", ".wav", ".flac", ".ogg", ".m4a"]:
                meta = extract_audio_metadata(file_path)
                results[filename] = {"type": "audio", "metadata": meta}
            elif ext == ".txt":
                meta = extract_txt_metadata(file_path)
                results[filename] = {"type": "text", "metadata": meta}
            elif ext == ".docx":
                meta = extract_docx_metadata(file_path)
                results[filename] = {"type": "docx", "metadata": meta}
            elif ext == ".doc":
                meta = extract_doc_metadata(file_path)
                results[filename] = {"type": "doc", "metadata": meta}
            elif ext in [".html", ".htm"]:
                meta = extract_html_metadata(file_path)
                results[filename] = {"type": "html", "metadata": meta}
            else:
                results[filename] = {
                    "type": "unknown",
                    "metadata": {"error": "Unsupported file type"},
                }
    return results


if __name__ == "__main__":
    folder_path = input("Enter the folder path containing files: ")
    metadata_results = process_files_in_folder(folder_path)
    with open("metadata_output.json", "w", encoding="utf-8") as outfile:
        json.dump(metadata_results, outfile, indent=4)
    print("Metadata extraction complete. Results saved in metadata_output.json")
