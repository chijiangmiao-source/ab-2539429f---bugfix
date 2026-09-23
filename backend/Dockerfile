# syntax=docker/dockerfile:1

# ---- runtime image -------------------------------------------------------
FROM python:3.11-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    API_HOST=0.0.0.0 \
    API_PORT=8000

WORKDIR /srv

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app

EXPOSE 8000

# slim has no curl; use the interpreter for the container health check
HEALTHCHECK --interval=10s --timeout=3s --start-period=5s --retries=5 \
    CMD python -c "import json,os,urllib.request,sys; \
r=urllib.request.urlopen('http://127.0.0.1:%s/healthz' % os.environ.get('API_PORT','8000'), timeout=3); \
sys.exit(0 if json.load(r)['status']=='ok' else 1)"

# shell form so $API_PORT is expanded at container start (configurable port)
CMD uvicorn app.main:app --host "$API_HOST" --port "$API_PORT"

# ---- test image used by the one-shot verify service ----------------------
FROM runtime AS test

COPY requirements-test.txt .
RUN pip install --no-cache-dir -r requirements-test.txt

COPY pytest.ini .
COPY tests ./tests

CMD ["python", "-m", "pytest", "-v"]
