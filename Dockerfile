FROM jupyter/minimal-notebook

RUN git clone https://github.com/hpullen/scikit-rt && \
    pip install -e scikit-rt && \
    mkdir ${HOME}/workdir

ENV JUPYTER_ENABLE_LAB=yes

WORKDIR "${HOME}/workdir"
