# Personal HR Benefits Assistant

A Streamlit capstone demonstration of a five-role HR benefits workflow using retrieval-augmented generation (RAG), metadata-filtered policy search, selective Tree-of-Thought-style evidence exploration, deterministic checks, and human escalation.

> **Demo only:** The included employee records and benefit policies are synthetic. Do not add real employee data, credentials, or confidential HR documents to this repository. The local demo checkbox is not authentication and this application is not approved for production HR use.

## Features

- Five coordinated roles: Coordinator/Controller, Policy Research, Employee Data, Thought Generator, and Evaluation/Validation.
- Policy ingestion for PDF, DOCX, TXT, HTML, and HTM.
- Policy metadata filtering by status, policy year, effective date, and expiration date **before** relevance ranking.
- TF-IDF retrieval with source filename and page information where available.
- Fresh CSV reads for each synthetic employee lookup.
- Selective comparison branches (up to three candidates, retaining the two strongest evidence paths).
- Deterministic checks for the sample full-time/30-hour eligibility rule when the necessary employee data is present.
- Potential policy-conflict detection and HR escalation when evidence is insufficient or inconsistent.
- Optional LangChain/OpenAI response generation; evidence-only fallback when no API key is configured.
- Refresh button and automatic policy index rebuild when files change.

## Requirements

- Python 3.10+
- Visual Studio Code (recommended)

## Run locally

From the project root (`hr_benefits_assistant`):

### Windows PowerShell

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```


Open the local URL printed by Streamlit (usually `http://localhost:8501`).

## Optional OpenAI integration

Without an API key, the application uses a conservative evidence-only response. To enable optional LangChain/OpenAI responses, set your key in the terminal before starting Streamlit.


## Sample data

- `data/employees.csv`: synthetic employee IDs E1001, E1002, and E1003.
- `data/policies/`: sample 2026 policy documents and an expired 2025 medical policy for testing.
- Add supported policy files under `data/policies/` (subfolders are scanned).

Policy documents should begin with metadata in this format:

```text
Policy Name: Example Medical Policy
Policy Year: 2026
Status: Active
Effective Date: 2026-01-01
Expiration Date: 2026-12-31
```

Documents without explicit metadata use conservative parser defaults in this demo; for reliable policy governance, add and verify all metadata before use. The **Refresh HR Data** button clears the cached index. File additions, edits, and removals also change the file signature and trigger a rebuild on the next policy query. The employee CSV is read fresh for every lookup.

## Security and production limitations

This capstone does **not** implement corporate SSO, verified identity, production role-based access control, encryption/key management, secure HRIS integration, immutable audit logging, or approved policy publishing. Before any real employee data is used, an organization must implement SSO, server-side authorization (employees may access only their own records; HR access must follow assigned roles), least-privilege service accounts, encryption, audit logs, secure secrets management, approved policy-document controls, and security/privacy testing.

The assistant is informational only. It does not change employee records, submit enrollment elections, or make binding HR determinations. Conflicting, missing, or uncertain information must be referred to HR. TF-IDF is used as a lightweight local retrieval baseline; it is not a production vector database or embedding pipeline.
