FROM python:3.12.3

# Fix: https://github.com/hadolint/hadolint/wiki/DL4006
# Fix: https://github.com/koalaman/shellcheck/wiki/SC3014
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

COPY --chown="${NB_UID}:${NB_GID}" nvidia-lib-dirs.sh "${CONDA_DIR}/etc/conda/activate.d/"

# https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/docker-specialized.html#dockerfiles
ENV NVIDIA_VISIBLE_DEVICES="all" \
    NVIDIA_DRIVER_CAPABILITIES="compute,utility"

ENV PYTHONUNBUFFERED=1
WORKDIR /app
COPY requirements.txt /app
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --upgrade --no-cache-dir --no-deps unsloth unsloth-zoo
COPY . /app
RUN chmod +x entrypoint.sh

RUN mkdir models

EXPOSE 8000
CMD ./entrypoint.sh