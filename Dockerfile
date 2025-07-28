FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

# 1. 필수 패키지 설치
RUN apt update && apt install -y \
    software-properties-common \
    wget \
    curl \
    gnupg \
    python3-pip \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# 2. OpenJDK 21 설치 (Adoptium 사용)
ENV JAVA_VERSION=21
ENV JAVA_HOME=/opt/java/openjdk

RUN mkdir -p "$JAVA_HOME" && \
    curl -L https://download.java.net/java/GA/jdk21/fd2272bbf8e04c3dbaee13770090416c/35/GPL/openjdk-21_linux-x64_bin.tar.gz \
    | tar -xz -C "$JAVA_HOME" --strip-components=1

ENV PATH="$JAVA_HOME/bin:$PATH"

# RUN mkdir -p "$JAVA_HOME" && \
#     wget https://download.java.net/java/GA/jdk21/fd2272bbf8e04c3dbaee13770090416c/35/GPL/openjdk-21_linux-x64_bin.tar.gz
#     tar -xz -C "$JAV A"
# RUN sudo wget https://download.java.net/java/GA/jdk21/fd2272bbf8e04c3dbaee13770090416c/35/GPL/openjdk-21_linux-x64_bin.tar.gz
# sudo tar xvf openjdk-21_linux-x64_bin.tar.gz

# RUN apt-get install openjdk-21-jdk

RUN java -version && python3 --version && pip3 --version

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY beir ./beir
RUN pip install -e beir

CMD [ "bash" ]
