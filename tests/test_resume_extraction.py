from pathlib import Path

from src.program import extract_text_from_file


def test_extract_text_from_txt_file(tmp_path):
    resume = tmp_path / "resume.txt"
    resume.write_text("Python developer with JavaScript and SQL experience.\n", encoding="utf-8")

    assert "Python developer" in extract_text_from_file(str(resume))
    assert "JavaScript" in extract_text_from_file(str(resume))


def test_extract_text_from_docx_file(tmp_path):
    from docx import Document

    resume = tmp_path / "resume.docx"
    doc = Document()
    doc.add_paragraph("Machine learning engineer with Python, TensorFlow, and Azure.")
    doc.save(resume)

    text = extract_text_from_file(str(resume))
    assert "Machine learning engineer" in text
    assert "TensorFlow" in text
