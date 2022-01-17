FROM jupyter/minimal-notebook

# Install scikit-rt.
RUN git clone https://github.com/scikit-rt/scikit-rt && \
    pip install -e scikit-rt && \
    mkdir ${HOME}/workdir

# Install elastix.
ARG ELASTIX_VERSION="5.0.1"
ARG ELASTIX_TARBALL="elastix-${ELASTIX_VERSION}-linux.tar.bz2"
RUN wget https://github.com/SuperElastix/elastix/releases/download/${ELASTIX_VERSION}/${ELASTIX_TARBALL} && \
    tar -xf ${ELASTIX_TARBALL} && \
    rm ${ELASTIX_TARBALL}

# Set up environment for running elastix.
ARG ELASTIX_DIR="${HOME}/elastix-${ELASTIX_VERSION}-linux"
ENV PATH="${ELASTIX_DIR}/bin:${PATH}"
ENV LD_LIBRARY_PATH="${ELASTIX_DIR}/lib"

ENV JUPYTER_ENABLE_LAB="yes"

WORKDIR "${HOME}/workdir"

LABEL "org.opencontainers.image.description"="Toolkit for radiotherapy image analysis"
LABEL "org.opencontainers.image.documentation"="https://scikit-rt.github.io/scikit-rt/"
LABEL "org.opencontainers.image.source"="https://github.com/scikit-rt/scikit-rt"
