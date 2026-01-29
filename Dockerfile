FROM python:3.11-slim

# Basic OS packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /work

# Install Python deps
RUN pip install --no-cache-dir -U pip && \
    pip install --no-cache-dir \
      numpy pandas scikit-image tifffile \
      torch torchvision \
      pytest
RUN pip install --no-cache-dir \
      requests
RUN pip install --no-cache-dir \
      anndata

# Make src/ importable without installing the package
ENV PYTHONPATH=/work/src

CMD ["bash"]
