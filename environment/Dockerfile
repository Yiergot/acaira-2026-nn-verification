# Mirrors the Google Colab environment closely enough to test the notebooks:
# x86_64 Linux, Python 3.12 system interpreter, and the same bootstrap script that the
# notebooks run to obtain a Python 3.11 virtualenv with Vehicle + Marabou.
FROM --platform=linux/amd64 python:3.12-slim

RUN apt-get update \
 && apt-get install -y --no-install-recommends curl ca-certificates git time \
 && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir uv jupyter nbconvert nbformat matplotlib pandas onnxruntime onnx "numpy<2" idx2numpy

ENV VENV=/content/venv
COPY colab_bootstrap.sh /opt/colab_bootstrap.sh
RUN bash /opt/colab_bootstrap.sh

ENV PATH="/content/venv/bin:${PATH}"
WORKDIR /work
CMD ["bash"]
