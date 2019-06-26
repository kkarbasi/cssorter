FROM ubuntu:latest
WORKDIR /run
RUN \
  apt-get update && \
  apt-get install -y python2.7 python-dev python-pip python-virtualenv && \
  rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8888
CMD ["jupyter", "notebook", "--port=8888", "--no-browser", "--ip=0.0.0.0", "--allow-root"]
