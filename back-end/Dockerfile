FROM ubuntu:22.04

# RUN echo "Acquire::Check-Valid-Until false;" > /etc/apt/apt.conf.d/99disable-valid-until
# RUN echo "APT::Get::AllowUnauthenticated true;" >> /etc/apt/apt.conf

# RUN apt-key adv --keyserver keyserver.ubuntu.com --recv-keys 3B4FE6ACC0B21F32

# RUN apt-get update && apt-get install -y sudo gnupg2 gnupg1
# RUN apt-get install -y ubuntu-keyring

# RUN rm -f /etc/apt/apt.conf.d/99disable-valid-until
# RUN sed -i '/APT::Get::AllowUnauthenticated/d' /etc/apt/apt.conf

# RUN adduser --disabled-password --gecos '' appuser

# RUN adduser appuser sudo

COPY . /app
# Set the working directory in the container to /front-end
WORKDIR /app

# RUN chown -R appuser:appuser /app

# USER appuser

VOLUME [ "/app" ]
# Install python

#RUN curl https://repo.waydro.id/waydroid.gpg | sudo dd of=/etc/apt/trusted.gpg.d/waydroid.gpg
#RUN apt-get install sudo

RUN apt-get update
RUN apt-get install -y python3 python3-pip libgirepository1.0-dev git libcairo2-dev
# RUN python3 -m venv /opt/venv
RUN python3 -m pip install --upgrade pip
COPY requirements.txt /app/requirements.txt
RUN pip install -r /app/requirements.txt

EXPOSE 8000
# RUN /venv/bin/activate
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0"]