FROM php:8.3-cli

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-venv python3-distutils python3-full \
    git unzip curl cloc docker.io \
    libxml2-dev libzip-dev zlib1g-dev pkg-config \
    libicu-dev libonig-dev libpq-dev libsqlite3-dev libpng-dev libjpeg-dev libfreetype6-dev \
    libxslt1-dev libmcrypt-dev libreadline-dev libssl-dev libcurl4-openssl-dev \
    && docker-php-ext-install \
        intl \
        mbstring \
        xml \
        zip \
        pdo \
        pdo_mysql \
        pdo_pgsql \
        opcache \
        xsl \
    && pecl install ast \
    && docker-php-ext-enable ast \
    && rm -rf /var/lib/apt/lists/*

RUN echo "memory_limit = 6G" > /usr/local/etc/php/conf.d/memory-limit.ini

COPY --from=composer:2 /usr/bin/composer /usr/local/bin/composer
ENV PATH="$PATH:/root/.composer/vendor/bin"
RUN composer global require vimeo/psalm phpstan/phpstan phan/phan

WORKDIR /app

RUN python3 -m venv venv
ENV PATH="/app/venv/bin:$PATH"

COPY collect.py /app/collect.py
COPY results.py /app/results.py

RUN /app/venv/bin/pip install --no-cache-dir requests urllib3 pandas numpy matplotlib scipy statsmodels lowess seaborn

VOLUME ["/app/workspace"]

ENTRYPOINT []
CMD ["bash"]
