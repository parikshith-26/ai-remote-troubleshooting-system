# Streamlit Audio Processing App

This project processes audio recordings from `recordings/`, generates translated transcripts in `transcripts/`, and saves AI troubleshooting responses in `responses/`.

## Run the Streamlit app

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Set your Groq API key:

- In PowerShell:

```powershell
$env:GROQ_API_KEY="your_groq_api_key"
```

- In Windows CMD:

```cmd
set GROQ_API_KEY=your_groq_api_key
```

3. Start the app:

```bash
streamlit run streamlit_app.py
```

## Notes

- Audio files should be placed in the `recordings/` folder.
- Transcripts are saved to `transcripts/`.
- AI responses are saved to `responses/`.
