# web-pqb

Web-PQB / Web-PRM experimental workspace for web-agent process quality annotation and scoring.

## Included

- Streamlit annotation app: `annotation_app.py`
- Canonical data loader: `canonical_data.py`
- Dataset conversion scripts: `scripts/`
- VisualWebArena render parser: `vwa_render_parser.py`
- Draft paper assets: `pdf.txt`, `main (7).pdf`, `想法.docx`, `标注培训准则.md`

## Not Included In Git

Large local artifacts are intentionally ignored:

- raw and processed datasets under `data/`
- annotation outputs under `annotations/`
- logs under `logs/`

## Quick Start

```bash
streamlit run annotation_app.py --server.headless true --server.port 8765
```
