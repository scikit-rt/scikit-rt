FROM jupyter/minimal-notebook

RUN git clone https://github.com/scikit-rt/scikit-rt && \
    pip install -e scikit-rt && \
    mkdir ${HOME}/workdir

ENV JUPYTER_ENABLE_LAB=yes

WORKDIR "${HOME}/workdir"

LABEL "org.opencontainers.image.description"="Toolkit for radiotherapy image analysis"
LABEL "org.opencontainers.image.documentation"="https://scikit-rt.github.io/scikit-rt/"
LABEL "org.opencontainers.image.source"="https://github.com/scikit-rt/scikit-rt"
