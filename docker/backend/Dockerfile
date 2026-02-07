FROM nvcr.io/nvidia/pytorch:25.09-py3

# Set CUDA environment variables
ENV CUDA_HOME=/usr/local/cuda
ENV CUDA_PATH=$CUDA_HOME
ENV PATH=$CUDA_HOME/bin:$PATH
ENV LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
ENV C_INCLUDE_PATH=$CUDA_HOME/include
ENV CPLUS_INCLUDE_PATH=$CUDA_HOME/include

# Install triton from source for latest blackwell support
RUN git clone https://github.com/triton-lang/triton.git && \
    cd triton && \
    git checkout c5d671f91d90f40900027382f98b17a3e04045f6 && \
    pip install -r python/requirements.txt && \
    pip install . && \
    cd ..

# Install xformers from source for blackwell support
RUN git clone --depth=1 https://github.com/facebookresearch/xformers --recursive && \
    cd xformers && \
    export TORCH_CUDA_ARCH_LIST="12.1" && \
    python setup.py install && \
    cd ..

# Install unsloth and other dependencies
RUN pip install --no-deps unsloth unsloth_zoo && \
    pip install bitsandbytes>=0.45.0 transformers==4.56.2 trl==0.22.2 nvidia-ml-py && \
    pip install datasets==4.3.0 accelerate peft packaging ninja sentencepiece diffusers hf_transfer tyro msgspec cut_cross_entropy


ENV PYTHONUNBUFFERED=1
WORKDIR /app
COPY requirements.txt /app

RUN pip install --no-cache-dir -r requirements.txt

RUN python -m spacy download en_core_web_sm
RUN crawl4ai-setup
COPY . /app
RUN chmod +x entrypoint.sh

RUN mkdir models

EXPOSE 8000
#CMD ["python", "-m", "http.server"]
CMD ./entrypoint.sh


