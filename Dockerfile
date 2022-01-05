ARG BASE_CONTAINER=jupyter/minimal-notebook
FROM $BASE_CONTAINER

USER ${NB_UID}

RUN git clone https://github.com/hpullen/scikit-rt && \
    cd scikit-rt && \
    pip install -e . && \
    mkdir ${HOME}/notebooks

ENV JUPYTER_ENABLE_LAB=yes

WORKDIR "${HOME}/notebooks"
