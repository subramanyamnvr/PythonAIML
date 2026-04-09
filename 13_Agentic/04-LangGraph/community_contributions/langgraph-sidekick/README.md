# langgraph-sidekick

A multi-agent AI assistant built with LangGraph: research, plan, execute tasks, and interact with the web and file system. Community contribution by [Tiago Iesbick](https://github.com/TiagoIesbick).

**Features:** Clarifier, planner, researcher, executor, summarizer, evaluator, finalizer ?? Web (Playwright) and file operations ?? Code execution (Python REPL) ?? Web search & Wikipedia ?? Optional WhatsApp ?? SQLite memory ?? Gradio UI ?? Side-effect approval for irreversible actions.

---

## Run from the repo root (agents)

**Prerequisites:** Python 3.13+, OpenAI API key.

From the repository root (`agents`):

```bash
cd 04-LangGraph/community_contributions/langgraph-sidekick
uv sync
playwright install chromium
```

Create a `.env` in `langgraph-sidekick` with at least:

```env
OPENAI_API_KEY=your_openai_api_key_here
TWILIO_ACCOUNT_SID=your_twilio_account_sid  # Optional, for WhatsApp
TWILIO_AUTH_TOKEN=your_twilio_auth_token     # Optional, for WhatsApp
TWILIO_WHATSAPP_FROM=your_whatsapp_number    # Optional, for WhatsApp
```

Start the app (still from the `langgraph-sidekick` folder):

```bash
uv run app.py
```

The Gradio UI opens in your browser.

---

## Config & troubleshooting

- **Model:** Default is `gpt-4o-mini`. Change in `sidekick.py` in the `ChatOpenAI` call.
- **Sandbox:** File operations use a `sandbox` directory (created automatically if missing).
- **Playwright:** If Chromium fails, see [Playwright Python docs](https://playwright.dev/python/docs/intro) for system deps.
- **API key:** App won't start without `OPENAI_API_KEY` in `.env`.
