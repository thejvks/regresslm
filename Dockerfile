FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY pyproject.toml ./
COPY src ./src
COPY examples ./examples
RUN pip install --no-cache-dir -e .

# Default: run the test suite. Override to run `regresslm ...` in CI.
CMD ["pytest", "-q"]
