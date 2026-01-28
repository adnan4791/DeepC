FROM gcc:latest

WORKDIR /app

# Install build dependencies if necessary (gcc image has most things)
# RUN apt-get update && apt-get install -y ...

# Keep container running
CMD ["tail", "-f", "/dev/null"]
