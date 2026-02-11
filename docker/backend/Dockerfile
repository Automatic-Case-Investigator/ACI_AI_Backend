FROM nvcr.io/nvidia/pytorch:25.09-py3

# Set CUDA environment variables
ENV CUDA_HOME=/usr/local/cuda
ENV CUDA_PATH=$CUDA_HOME
ENV PATH=$CUDA_HOME/bin:$PATH
ENV LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
ENV C_INCLUDE_PATH=$CUDA_HOME/include
ENV CPLUS_INCLUDE_PATH=$CUDA_HOME/include

ENV PYTHONUNBUFFERED=1
WORKDIR /app
COPY requirements.txt /app

RUN --mount=type=cache,target=/root/.cache/pip pip install -r requirements.txt

RUN python -m spacy download en_core_web_sm
RUN crawl4ai-setup
COPY . /app
RUN chmod +x entrypoint.sh

RUN mkdir models

EXPOSE 8000
CMD ./entrypoint.sh


