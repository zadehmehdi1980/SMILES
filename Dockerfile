# Mehdi Zadeh (Nimahm@gmail.com)
# How to write a Docker file : https://pythonspeed.com/articles/activate-conda-dockerfile/

FROM continuumio/miniconda3:latest 

# Create the environment:
# COPY environment.yml .

ADD . /code
WORKDIR /code
RUN conda env create -f environment.yml

# Make RUN commands use the new environment:
SHELL ["conda", "run", "-n", "myenvminiconda", "/bin/bash", "-c"]

# Make sure the environment is activated:
RUN echo "Make sure flask is installed:"
RUN python -c "import flask"

#Update Linux packages (If needed)

RUN echo 'debconf debconf/frontend select Noninteractive' | debconf-set-selections
RUN apt-get update && apt-get install -y -q && apt-get upgrade -y && apt-get install build-essential -y && apt-get install unzip -y &&  apt-get install git-core -y

#Define a variable (if needed in any case)
ENV INPUT_TEXT 'Cc1ccc(/C=C2\C(=O)NC(=O)N(Cc3ccccc3Cl)C2=O)o1'

# Path (if needed in any case)
ENV PATH=$PATH:/fingerprintSMILES

# Flask Port (Default port) 

EXPOSE 5000


#Run Entry point with Gunicorn to handle multiple concurrent requests and in Miniconda Env

CMD ["conda", "run", "-n", "myenvminiconda", "gunicorn"  ,"wsgi:app", "--bind", "0.0.0.0:5000"]



