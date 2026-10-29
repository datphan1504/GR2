
FROM golang:1.24.1 AS builder-usque
RUN apt-get update && apt-get install -y git
WORKDIR /app
RUN git clone https://github.com/Diniboy1123/usque.git .
RUN git checkout d0eb96e
RUN go mod download
RUN go build -o usque -ldflags="-s -w" .


FROM golang:1.24.1 AS builder-masque
RUN apt-get update && apt-get install -y git
WORKDIR /app
RUN git clone --branch v2.9.0 --single-branch https://github.com/ircfspace/masque-plus.git .
RUN go mod download
RUN go build -o masque-plus .


FROM ubuntu:22.04


ENV DEBIAN_FRONTEND=noninteractive


RUN apt-get update && apt-get install -y \
    curl \
    git \
    vim \
    && rm -rf /var/lib/apt/lists/*


WORKDIR /workspace


COPY --from=builder-usque /app/usque /workspace//usque
COPY --from=builder-masque /app/masque-plus /workspace//masque-plus
RUN chmod +x /workspace/usque /workspace/masque-plus


COPY ./copy_folder/ /workspace/


RUN chmod +x /workspace/container_install.sh
RUN ./container_install.sh

CMD ["tail", "-f", "/dev/null"]