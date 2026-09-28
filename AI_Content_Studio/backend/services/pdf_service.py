import io
import fitz

def extract_pdf_text(pdf_bytes: bytes):
    document = fitz.open(
        stream=io.BytesIO(pdf_bytes),
        filetype="pdf"
    )
    pages = []
    full_text = []
    for page_number, page in enumerate(document, start=1):
        text = page.get_text("text").strip()
        pages.append({
            "page": page_number,
            "text": text
        })
        if text:
            full_text.append(text)
    document.close()

    return {
        "pages": len(pages),
        "text": "\n\n".join(full_text),
        "page_data": pages
    }