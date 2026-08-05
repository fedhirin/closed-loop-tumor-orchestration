FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-runtime
WORKDIR /workspace
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
RUN pip install --no-deps -e .
ENTRYPOINT ["tumor-train"]

