FROM python:3.12.3

ENV PYTHONUNBUFFERED=1
WORKDIR /app
COPY requirements.txt /app
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --upgrade --no-cache-dir --no-deps unsloth unsloth-zoo
COPY . /app
RUN mkdir models

# Fix: https://github.com/hadolint/hadolint/wiki/DL4006
# Fix: https://github.com/koalaman/shellcheck/wiki/SC3014
SHELL ["/bin/bash", "-o", "pipefail", "-c"]


COPY --chown="${NB_UID}:${NB_GID}" nvidia-lib-dirs.sh "${CONDA_DIR}/etc/conda/activate.d/"

# https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/docker-specialized.html#dockerfiles
ENV NVIDIA_VISIBLE_DEVICES="all" \
    NVIDIA_DRIVER_CAPABILITIES="compute,utility"

EXPOSE 8000
CMD python3 manage.py makemigrations && python3 manage.py migrate && python3 manage.py runserver 0.0.0.0:8000