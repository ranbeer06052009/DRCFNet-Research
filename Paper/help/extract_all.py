import os
import sys

def extract_pdf_to_txt(pdf_path, txt_path):
    print(f"[*] Extracting: {os.path.basename(pdf_path)} -> {os.path.basename(txt_path)}")
    # Try pypdf
    try:
        import pypdf
        reader = pypdf.PdfReader(pdf_path)
        text = ""
        for i, page in enumerate(reader.pages):
            text += f"\n--- PAGE {i+1} ---\n"
            page_text = page.extract_text()
            if page_text:
                text += page_text
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(text)
        print("    [pypdf] Extraction successful.")
        return True
    except Exception as e:
        print(f"    [pypdf] Failed: {e}")
        
    # Try PyMuPDF (fitz)
    try:
        import fitz
        doc = fitz.open(pdf_path)
        text = ""
        for i, page in enumerate(doc):
            text += f"\n--- PAGE {i+1} ---\n"
            text += page.get_text()
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(text)
        print("    [fitz] Extraction successful.")
        return True
    except Exception as e:
        print(f"    [fitz] Failed: {e}")
        
    # Try pdfplumber
    try:
        import pdfplumber
        text = ""
        with pdfplumber.open(pdf_path) as pdf:
            for i, page in enumerate(pdf.pages):
                text += f"\n--- PAGE {i+1} ---\n"
                page_text = page.extract_text()
                if page_text:
                    text += page_text
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(text)
        print("    [pdfplumber] Extraction successful.")
        return True
    except Exception as e:
        print(f"    [pdfplumber] Failed: {e}")
        
    # Try pdfminer
    try:
        from pdfminer.high_level import extract_text
        text = extract_text(pdf_path)
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(text)
        print("    [pdfminer] Extraction successful.")
        return True
    except Exception as e:
        print(f"    [pdfminer] Failed: {e}")
        
    return False

def main():
    help_dir = r"d:\DRCFNet\Paper\help"
    files = os.listdir(help_dir)
    pdf_files = [f for f in files if f.lower().endswith('.pdf')]
    print(f"[*] Found {len(pdf_files)} PDF files in help directory.")
    
    for pdf_file in pdf_files:
        pdf_path = os.path.join(help_dir, pdf_file)
        txt_file = pdf_file[:-4] + "_text.txt"
        txt_path = os.path.join(help_dir, txt_file)
        
        success = extract_pdf_to_txt(pdf_path, txt_path)
        if not success:
            print(f"[!] Could not extract {pdf_file} using any available parser.")

if __name__ == "__main__":
    main()
