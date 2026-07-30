FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive \
    CICY_DOCKER_BUILD=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates curl git ffmpeg bzip2 build-essential \
      libgl1 libglib2.0-0 libsndfile1 fonts-noto-core fonts-noto-extra fonts-noto-cjk \
    && rm -rf /var/lib/apt/lists/*

COPY scripts/provision/mt/provision.sh /opt/cicy-koubo/mt-provision.sh
COPY scripts/provision/cosy/provision.sh /opt/cicy-koubo/cosy-provision.sh
COPY scripts/provision/cosy/cosyvoice_tts.py /opt/cicy-koubo/cosyvoice_tts.py
COPY docker/gpu/entrypoint.sh /opt/cicy-koubo/entrypoint.sh

RUN chmod +x /opt/cicy-koubo/*.sh \
    && bash /opt/cicy-koubo/mt-provision.sh \
    && bash /opt/cicy-koubo/cosy-provision.sh \
    && cp /opt/cicy-koubo/cosyvoice_tts.py /content/cosy/cosyvoice_tts.py \
    && rm -rf /content/.cache/pip /root/.cache/pip

ENV CICY_DOCKER_BUILD=0
ENTRYPOINT ["/opt/cicy-koubo/entrypoint.sh"]
CMD ["status"]
