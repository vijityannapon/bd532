"""Build the term-project report PDF from docs/*.md (Sections 1-4).

    python scripts/build_report.py --author "ชื่อ นามสกุล" --student-id 6XXXXXXXX \
        --github https://github.com/<user>/bd532-credit-risk

Pipeline: Markdown -> HTML (figures numbered, Mermaid diagrams rendered by
mermaid.js) -> PDF via headless Google Chrome -> page numbers stamped with pypdf.
Requires: pip install -r requirements-report.txt  and Google Chrome.
"""
from __future__ import annotations

import argparse
import html
import json
import re
import shutil
import subprocess
from datetime import date
from io import BytesIO
from pathlib import Path

import markdown
from pypdf import PdfReader, PdfWriter
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
OUT_DIR = ROOT / "report"
SECTIONS = ["01_project_proposal.md", "02_system_design.md", "03_ml_implementation.md",
            "04_ui_model_interface.md"]
CHROME_CANDIDATES = [
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "google-chrome", "google-chrome-stable", "chromium", "chromium-browser",
]

CSS = """
@page { size: A4; margin: 18mm 17mm 20mm 17mm; }
* { box-sizing: border-box; }
body { font-family: 'Sarabun', 'Thonburi', sans-serif; font-size: 12pt; line-height: 1.5; color: #1d2330; }
h1 { font-size: 21pt; color: #1f3a8a; border-bottom: 2px solid #1f3a8a; padding-bottom: 4px;
     margin: 0 0 12px; page-break-before: always; }
h2 { font-size: 15.5pt; color: #1f3a8a; margin: 18px 0 6px; page-break-after: avoid; }
h3 { font-size: 13.5pt; margin: 14px 0 4px; page-break-after: avoid; }
p, li { text-align: left; }
ul, ol { margin: 4px 0 8px; padding-left: 22px; }
li { margin: 2px 0; }
table { border-collapse: collapse; width: 100%; margin: 8px 0 12px; font-size: 10.5pt; line-height: 1.35;
        page-break-inside: auto; }
tr { page-break-inside: avoid; }
th { background: #eef2fb; text-align: left; }
th, td { border: 1px solid #c9d1e3; padding: 4px 6px; vertical-align: top; }
code { font-family: Menlo, Consolas, monospace; font-size: 9pt; background: #f3f4f6; padding: 0 3px; border-radius: 3px; }
pre { background: #f6f8fa; border: 1px solid #e3e6eb; border-radius: 6px; padding: 8px 10px; font-size: 8.6pt;
      line-height: 1.35; white-space: pre-wrap; word-break: break-word; page-break-inside: avoid; }
pre code { background: none; padding: 0; font-size: inherit; }
blockquote { border-left: 4px solid #1f3a8a; margin: 8px 0; padding: 4px 12px; background: #f5f7fc; color: #334; }
figure { margin: 10px 0 14px; text-align: center; page-break-inside: avoid; }
figure img { max-width: 100%; max-height: 100mm; border: 1px solid #e3e6eb; }
figcaption { font-size: 10.5pt; color: #475467; margin-top: 4px; }
.diagram { margin: 10px 0 14px; text-align: center; page-break-inside: avoid; }
.diagram svg { max-width: 100% !important; height: auto; max-height: 200mm; }
.diagram .cap { font-size: 10.5pt; color: #475467; margin-top: 4px; }
pre.mermaid { background: none; border: none; font-family: 'Sarabun', sans-serif; font-size: 11pt; }
.cover { height: 255mm; display: flex; flex-direction: column; justify-content: space-between; }
.cover .top { margin-top: 30mm; }
.cover .course { font-size: 14pt; color: #475467; letter-spacing: .5px; }
.cover h0 { display: block; font-size: 30pt; font-weight: 700; color: #1f3a8a; line-height: 1.25; margin: 14px 0 6px; }
.cover .sub { font-size: 16pt; color: #334; }
.cover .meta { font-size: 13pt; line-height: 1.9; border-top: 2px solid #1f3a8a; padding-top: 12px; }
.toc h1 { page-break-before: always; }
.toc ol { font-size: 13pt; line-height: 1.8; }
.toc ol ol { font-size: 11.5pt; line-height: 1.6; color: #475467; }
.summary { page-break-before: always; }
.kpis { display: flex; gap: 10px; margin: 10px 0 14px; }
.kpi { flex: 1; border: 1px solid #c9d1e3; border-radius: 8px; padding: 8px 10px; background: #f8faff; }
.kpi .v { font-size: 19pt; font-weight: 700; color: #1f3a8a; }
.kpi .l { font-size: 10pt; color: #475467; line-height: 1.3; }
"""

SUMMARY_MD = """
**CreditLens** คือระบบ AI ต้นแบบ (proof-of-concept) ที่ช่วยเจ้าหน้าที่สินเชื่อประเมินโอกาสผิดนัดชำระ (Probability of Default, PD) ของผู้ขอสินเชื่อรายย่อย
พร้อมเหตุผลที่อธิบายได้ด้วย SHAP ระบบ **ไม่ตัดสินใจแทนมนุษย์** แต่จัดคำขอเป็น 3 ระดับ (LOW_RISK / REVIEW / HIGH_RISK) เพื่อให้เจ้าหน้าที่โฟกัสเคสที่เสี่ยงจริง

* **ข้อมูล:** Home Credit Default Risk (Kaggle) 307,511 คำขอ อัตราผิดนัด 8.07%
* **โมเดล:** เปรียบเทียบ 6 versions ใน MLflow (Logistic Regression, Random Forest, HistGradientBoosting และ fairness reweighing) แล้วเลือกด้วยเกณฑ์ calibration → ROC-AUC → fairness → latency
* **ผล:** โมเดลที่เลือก (`v6_hgb_tuned_reweighed`) ได้ test ROC-AUC 0.764, KS 0.399 และ PD ที่ calibrated ส่วน reweighing ลดช่องว่าง FPR ระหว่างเพศลง 63% โดยไม่เสียความแม่นยำ
* **ระบบ:** FastAPI REST API (5 endpoints) + Web UI ภาษาไทยที่มี feedback loop ผ่าน 40 unit tests และ interface test 17/17 scenarios (Postman)
* **ความเสี่ยง:** FTA แสดงว่าไม่มี single-point failure (18 minimum cut sets ขนาด 2) เพราะออกแบบให้มี human-in-the-loop
"""


def kpi_html(results: dict) -> str:
    t = results["test_metrics"]
    bands = results["decision_bands_test"]
    g = results["fairness_test"]["gender"]["summary"]
    items = [(f"{t['roc_auc']:.3f}", "Test ROC-AUC"), (f"{t['ks']:.3f}", "KS statistic"),
             (f"{bands['HIGH_RISK']['actual_default_rate'] / bands['LOW_RISK']['actual_default_rate']:.1f}×",
              "Default rate HIGH vs LOW band"),
             (f"{g['disparate_impact_ratio']:.2f}", "Gender disparate impact (≥ 0.8)"),
             ("17/17", "Interface tests passed")]
    return '<div class="kpis">' + "".join(
        f'<div class="kpi"><div class="v">{v}</div><div class="l">{l}</div></div>' for v, l in items) + "</div>"


LIST_ITEM = re.compile(r"^\s*(?:[*+-]|\d+\.)\s+")


def separate_lists(text: str) -> str:
    """GitHub renders a list that directly follows a paragraph line; python-markdown
    needs a blank line first. Insert one (outside fenced code blocks)."""
    out, in_fence = [], False
    for line in text.split("\n"):
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
        prev = out[-1] if out else ""
        if (not in_fence and LIST_ITEM.match(line) and prev.strip()
                and not LIST_ITEM.match(prev) and not prev.startswith((" ", "\t", "|"))):
            out.append("")
        out.append(line)
    return "\n".join(out)


def md_to_html(text: str) -> str:
    diagrams: list[str] = []

    def stash(m):  # keep Mermaid source away from the Markdown parser
        diagrams.append(m.group(1))
        return f"\n\nMERMAIDBLOCK{len(diagrams) - 1}\n\n"

    text = re.sub(r"```mermaid\n(.*?)```", stash, text, flags=re.S)
    text = separate_lists(text)
    out = markdown.markdown(text, extensions=["tables", "fenced_code", "sane_lists"])
    for i, src in enumerate(diagrams):
        out = out.replace(f"<p>MERMAIDBLOCK{i}</p>",
                          f'<div class="diagram"><pre class="mermaid">{html.escape(src)}</pre></div>')
    return out


def number_figures(body: str) -> str:
    counter = {"n": 0}

    def fig(m):
        counter["n"] += 1
        attrs = m.group(1)
        alt = re.search(r'alt="([^"]*)"', attrs)
        cap = html.unescape(alt.group(1)) if alt else ""
        return (f'<figure><img{attrs}><figcaption>รูปที่ {counter["n"]}: '
                f'{html.escape(cap)}</figcaption></figure>')

    return re.sub(r"<p>\s*<img([^>]*?)\s*/?>\s*</p>", fig, body)


def build_html(args) -> Path:
    results = json.loads((ROOT / "reports" / "results.json").read_text(encoding="utf-8"))
    parts, toc = [], []
    for name in SECTIONS:
        text = (DOCS / name).read_text(encoding="utf-8")
        h1 = re.search(r"^# (.+)$", text, flags=re.M).group(1)
        subs = re.findall(r"^## (.+)$", text, flags=re.M)
        toc.append((h1, subs))
        parts.append(md_to_html(text))
    body = number_figures("\n".join(parts))

    toc_html = "<ol>" + "".join(
        f"<li>{html.escape(h)}<ol>" + "".join(f"<li>{html.escape(s)}</li>" for s in subs) + "</ol></li>"
        for h, subs in toc) + "</ol>"
    cover = f"""
<section class="cover">
  <div class="top">
    <div class="course">BD532 · Term Project</div>
    <h0>CreditLens</h0>
    <div class="sub">ระบบ AI ช่วยพิจารณาความเสี่ยงสินเชื่อแบบอธิบายได้<br>
      Explainable AI-assisted Loan Default Risk Assessment</div>
  </div>
  <div class="meta">
    ผู้จัดทำ: {html.escape(args.author)}<br>
    รหัสนักศึกษา: {html.escape(args.student_id)}<br>
    Source code: {html.escape(args.github)}<br>
    วันที่: {args.date}
  </div>
</section>"""
    summary = (f'<section class="summary"><h1 style="page-break-before:avoid">บทสรุปผู้บริหาร (Executive Summary)</h1>'
               f'{kpi_html(results)}{markdown.markdown(SUMMARY_MD)}</section>')
    doc = f"""<!doctype html><html lang="th"><head><meta charset="utf-8">
<base href="{DOCS.as_uri()}/">
<title>BD532 Term Project Report - CreditLens</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Sarabun:wght@400;600;700&display=swap" rel="stylesheet">
<style>{CSS}</style></head><body>
{cover}
{summary}
<section class="toc"><h1>สารบัญ</h1>{toc_html}</section>
{body}
<script src="https://cdn.jsdelivr.net/npm/mermaid@10.9.1/dist/mermaid.min.js"></script>
<script>
  mermaid.initialize({{startOnLoad: false, theme: 'neutral', securityLevel: 'loose',
    fontFamily: 'Sarabun, sans-serif', flowchart: {{useMaxWidth: true, htmlLabels: true}},
    sequence: {{useMaxWidth: true}}}});
  // Render one by one with explicit unique ids: mermaid.run() derives ids from
  // Date.now(), which is frozen under headless virtual time -> all diagrams collide.
  (async () => {{
    await document.fonts.ready;
    const blocks = [...document.querySelectorAll('pre.mermaid')];
    for (let i = 0; i < blocks.length; i++) {{
      const {{svg}} = await mermaid.render('diagram' + i, blocks[i].textContent);
      const holder = document.createElement('div');
      holder.innerHTML = svg;
      blocks[i].replaceWith(holder.firstElementChild);
    }}
    document.body.setAttribute('data-ready', '1');
  }})();
</script>
</body></html>"""
    OUT_DIR.mkdir(exist_ok=True)
    out = OUT_DIR / "report.html"
    out.write_text(doc, encoding="utf-8")
    return out


def find_chrome() -> str:
    for c in CHROME_CANDIDATES:
        if Path(c).exists() or shutil.which(c):
            return c
    raise SystemExit("Google Chrome / Chromium not found - needed to render the PDF.")


def render_pdf(html_path: Path, pdf_path: Path) -> None:
    """Two passes: (1) let mermaid.js render and dump the final DOM, (2) print the
    static HTML. Printing directly can snapshot diagrams mid-render (they overlap)."""
    chrome = find_chrome()
    base = [chrome, "--headless=new", "--disable-gpu", "--run-all-compositor-stages-before-draw"]
    dom = subprocess.run(base + ["--virtual-time-budget=30000", "--dump-dom", html_path.as_uri()],
                         check=True, capture_output=True, timeout=180, text=True).stdout
    n_src = html_path.read_text(encoding="utf-8").count('class="mermaid"')
    n_svg = len(set(re.findall(r'<svg[^>]*id="(diagram\d+)"', dom)))
    if n_svg < n_src:
        raise SystemExit(f"Only {n_svg}/{n_src} Mermaid diagrams rendered - check network access to the CDN.")
    dom = re.sub(r"<script\b.*?</script>", "", dom, flags=re.S)  # static page: no re-rendering
    static = html_path.with_name("report_static.html")
    static.write_text("<!doctype html>\n" + dom, encoding="utf-8")
    subprocess.run(base + ["--no-pdf-header-footer", "--virtual-time-budget=10000",
                           f"--print-to-pdf={pdf_path}", static.as_uri()],
                   check=True, capture_output=True, timeout=180)


def stamp_page_numbers(pdf_path: Path, author: str) -> int:
    writer = PdfWriter(clone_from=str(pdf_path))
    total = len(writer.pages)
    for i, page in enumerate(writer.pages):
        if i > 0:  # no number on the cover
            buf = BytesIO()
            c = canvas.Canvas(buf, pagesize=A4)
            c.setFont("Helvetica", 9)
            c.setFillGray(0.45)
            c.drawCentredString(A4[0] / 2, 22, f"{i + 1} / {total}")
            c.drawString(48, 22, "BD532 Term Project - CreditLens")
            c.save()
            buf.seek(0)
            page.merge_page(PdfReader(buf).pages[0])
    writer.add_metadata({"/Title": "BD532 Term Project Report - CreditLens", "/Author": author})
    with pdf_path.open("wb") as f:
        writer.write(f)
    return total


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--author", default="[ชื่อ-นามสกุล]")
    ap.add_argument("--student-id", default="[รหัสนักศึกษา]")
    ap.add_argument("--github", default="[GitHub repository URL]")
    ap.add_argument("--date", default=date.today().strftime("%d/%m/%Y"))
    ap.add_argument("--out", type=Path, default=OUT_DIR / "BD532_TermProject_Report.pdf")
    args = ap.parse_args()
    html_path = build_html(args)
    render_pdf(html_path, args.out)
    pages = stamp_page_numbers(args.out, args.author)
    print(f"Wrote {args.out} ({pages} pages)")


if __name__ == "__main__":
    main()
