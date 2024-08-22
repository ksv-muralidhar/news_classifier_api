FROM python:3.10-slim
WORKDIR /code
COPY ./requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt
RUN apt update && apt install -y ffmpeg
RUN apt -y install wget
RUN apt -y install unzip

RUN apt-get install -y \
    gnupg \
    ca-certificates \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libfontconfig1 \
    libnss3 \
    libatk-bridge2.0-0 \
    libatk1.0-0 \
    libatspi2.0-0 \
    libcups2 \
    libcurl4 \
    libgtk-3-0 \
    libnspr4 \
    libxcomposite1 \
    libxdamage1 \
    xdg-utils \
    fonts-liberation \
    libu2f-udev \
    && rm -rf /var/lib/apt/lists/*

RUN wget https://dl.google.com/linux/direct/google-chrome-stable_current_amd64.deb && \
    dpkg -i google-chrome-stable_current_amd64.deb && \
    apt-get -f install -y && \
    rm google-chrome-stable_current_amd64.deb
    
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
	PATH=/home/user/.local/bin:$PATH \
    CHROMEDRIVERURL=https://storage.googleapis.com/chrome-for-testing-public/127.0.6533.119/linux64/chromedriver-linux64.zip \
    CHROMEDRIVERFILENAME=chromedriver-linux64.zip
    

WORKDIR $HOME/app

COPY --chown=user . $HOME/app

RUN wget -P $HOME/app $CHROMEDRIVERURL
RUN unzip $HOME/app/$CHROMEDRIVERFILENAME
RUN rm $HOME/app/$CHROMEDRIVERFILENAME

RUN chmod +x $HOME/app/chromedriver-linux64/chromedriver

RUN ls -ltr

EXPOSE 7860
ENTRYPOINT ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "3"]
