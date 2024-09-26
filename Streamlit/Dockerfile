FROM python:3.10-slim

# Install curl
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

COPY . /app
WORKDIR /app



RUN pip install -r requirements.txt

EXPOSE 80 8501 11434

RUN mkdir ~/.streamlit
RUN cp config.toml ~/.streamlit/config.toml
RUN cp credentials.toml ~/.streamlit/credentials.toml

COPY start.sh /start.sh
RUN chmod +x /start.sh

COPY start.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

CMD ["/start.sh"]
