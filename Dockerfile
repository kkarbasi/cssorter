FROM ubuntu:latest
WORKDIR /run
RUN \
  apt-get update && \
  apt-get install -y python3 python3-pip build-essential libssl-dev libffi-dev python3-dev python3-venv && \
  rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip3 install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8888
CMD ["jupyter", "notebook", "--port=8888", "--no-browser", "--ip=0.0.0.0", "--allow-root"]
