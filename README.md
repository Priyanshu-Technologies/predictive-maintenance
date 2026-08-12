# Predictive Maintenance API

A REST API that serves your trained XGBoost model from
`Industrial_Predictive_Maintenance.ipynb`. Send it live sensor readings
(temperature, speed, torque, tool wear), get back a failure probability and
risk level. Includes a small browser demo page so anyone can try it without
writing code.

## Project layout

```
pm-api/
├── app/
│   ├── main.py        # FastAPI app + routes
│   ├── model.py        # loads model/scaler, runs predictions
│   ├── features.py     # feature engineering (mirrors the notebook exactly)
│   ├── schemas.py       # request/response validation
│   └── config.py         # paths & constants
├── models/               # your trained artifacts (already copied in)
│   ├── best_model.pkl
│   ├── scaler.pkl
│   ├── feature_names.pkl
│   └── model_results.csv
├── static/
│   └── index.html        # browser demo UI
├── requirements.txt
├── Dockerfile
├── test_api.py            # smoke tests
└── README.md
```

`smote.pkl` from your project isn't used here on purpose — SMOTE is a
**training-time-only** technique for rebalancing classes before fitting. It
must never be applied to real data at prediction time, so it's not part of
the serving path.

## Run it locally

```bash
cd pm-api
python -m venv .venv && source .venv/bin/activate   # optional but recommended
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Then open:
- `http://127.0.0.1:8000/` — the demo page (form + live result)
- `http://127.0.0.1:8000/docs` — interactive API docs (Swagger UI), lets you try every endpoint
- `http://127.0.0.1:8000/health` — health check

Run the smoke tests while the server is running:
```bash
pip install requests
python test_api.py
```

## API reference

### `POST /predict`
Score one machine reading.

Request:
```json
{
  "type": "M",
  "air_temperature": 300.5,
  "process_temperature": 310.2,
  "rotational_speed": 1500,
  "torque": 40.5,
  "tool_wear": 110
}
```
`type` is the product quality variant used in the original dataset: `L` (low), `M` (medium), or `H` (high).

Response:
```json
{
  "prediction": 0,
  "label": "Normal",
  "failure_probability": 0.004136,
  "risk_level": "Low"
}
```

### `POST /predict/batch`
Same shape, but `{"readings": [ {...}, {...} ]}` — up to 500 at once. Returns `{"results": [...]}`.

### `GET /model/info`
Returns the deployed model's name, the exact feature list it expects, and the metrics comparison table from your notebook (all six models you trained, plus which one is live).

### `GET /health`
Basic liveness/readiness check — confirms the model loaded.

## Deploying it so anyone can use it

The easiest free/cheap options for a FastAPI + ML model app, roughly ordered by effort:

### Option A — Render.com (free tier, simplest)
1. Push this `pm-api/` folder to a new GitHub repo.
2. On [render.com](https://render.com) → New → Web Service → connect the repo.
3. Render auto-detects the `Dockerfile` — leave build/run commands blank, it'll use it.
4. Deploy. You'll get a public URL like `https://your-app.onrender.com`.
5. Note: the model artifacts (`models/*.pkl`) are ~260KB total, small enough to just commit to the repo — no need for external storage.

### Option B — Railway.app
Same idea as Render: connect the GitHub repo, Railway detects the Dockerfile and deploys automatically. Slightly more generous free tier at time of writing.

### Option C — Fly.io
```bash
fly launch      # detects the Dockerfile, asks a few questions
fly deploy
```

### Option D — Hugging Face Spaces (Docker SDK)
Good if you want it discoverable alongside other ML demos. Create a Space, choose "Docker" as the SDK, push this folder — it reads the same `Dockerfile`.

Whichever you choose, the pattern is identical: point the platform at this folder, let it build the `Dockerfile`, and it exposes `/` (demo page) and `/docs` (API) on a public URL.

### A note on the free tiers
Free tiers on Render/Railway/Fly typically spin the container down after a period of inactivity, so the first request after idle time can take 10-30 seconds while it wakes up. That's expected — not a bug in this code.

## Extending it later
- **Auth**: add an API key check in `app/main.py` (a `Depends()` that checks a header) before opening this up publicly for real traffic.
- **Rate limiting**: `slowapi` (a FastAPI-friendly wrapper around `limits`) is a drop-in option.
- **Retraining**: if you retrain the model, just drop the new `best_model.pkl` / `scaler.pkl` / `feature_names.pkl` into `models/` — nothing else needs to change, since `app/model.py` reindexes columns off `feature_names.pkl` rather than a hardcoded list.
- **Logging predictions**: for monitoring drift over time, log each request/response pair to a file or database inside the `/predict` route.
