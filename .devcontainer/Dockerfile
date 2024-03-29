FROM python:3.11

# Set up the working directory
WORKDIR /workspace

RUN apt-get update && \
    apt-get install -y locales && \
    sed -i -e 's/# en_US.UTF-8 UTF-8/en_US.UTF-8 UTF-8/' /etc/locale.gen && \
    dpkg-reconfigure --frontend=noninteractive locales && \
    update-locale LANG=en_US.UTF-8

# Create a dedicated virtual environment for Poetry
RUN python3 -m venv /opt/poetry-env

# Install Poetry in the dedicated virtual environment
RUN /opt/poetry-env/bin/pip install poetry

# Copy the Poetry files
COPY pyproject.toml poetry.lock ./

# Install dependencies
RUN /opt/poetry-env/bin/poetry config virtualenvs.create false \
    && /opt/poetry-env/bin/poetry install --no-root --no-interaction --no-ansi

# Copy the project files
COPY . .

# Set the default command to keep the container running
CMD ["sleep", "infinity"]