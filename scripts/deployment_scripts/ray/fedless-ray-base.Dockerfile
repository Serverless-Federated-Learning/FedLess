FROM rayproject/ray:2.2.0-py38
COPY fedless_requirements.txt .
RUN pip install --no-cache-dir -r fedless_requirements.txt