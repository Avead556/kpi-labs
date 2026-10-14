FROM python:3.14-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    LIBGL_ALWAYS_SOFTWARE=1 \
    GALLIUM_DRIVER=llvmpipe

RUN apt-get update && apt-get install -y --no-install-recommends \
        xvfb \
        libgl1 \
        libglx-mesa0 \
        libgl1-mesa-dri \
        libx11-6 \
        libxext6 \
        libxfixes3 \
        libxrandr2 \
        libxinerama1 \
        libxcursor1 \
        libxi6 \
        libxkbcommon0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY docker-entrypoint.sh /usr/local/bin/
COPY labkit/ ./labkit/
COPY web/ ./web/
COPY mv/ ./mv/
COPY ai/ ./ai/

RUN chmod +x /usr/local/bin/docker-entrypoint.sh \
    && mkdir -p /tmp/.X11-unix && chmod 1777 /tmp/.X11-unix \
    && useradd --create-home --uid 10001 hub && chown -R hub:hub /app
USER hub

EXPOSE 8000

ENTRYPOINT ["docker-entrypoint.sh"]
CMD ["python", "-m", "uvicorn", "web.app:app", "--host", "0.0.0.0", "--port", "8000"]
