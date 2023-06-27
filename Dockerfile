FROM jupyter/minimal-notebook

# Install specific version of python.
RUN conda install python=3.10

# Install scikit-rt.
RUN git clone https://github.com/scikit-rt/scikit-rt \
    && python -m pip install -e scikit-rt \
    && mkdir ${HOME}/workdir

# Copy example Jupyter notebooks.
COPY examples/notebooks/image*.ipynb ${HOME}/workdir/

# Install elastix.
ARG ELASTIX_VERSION="5.0.1"
ARG ELASTIX_TARBALL="elastix-${ELASTIX_VERSION}-linux.tar.bz2"
RUN wget https://github.com/SuperElastix/elastix/releases/download/${ELASTIX_VERSION}/${ELASTIX_TARBALL} \
    && tar -xf ${ELASTIX_TARBALL} \
    && rm ${ELASTIX_TARBALL}

# Set up environment for running elastix.
ARG ELASTIX_DIR="${HOME}/elastix-${ELASTIX_VERSION}-linux"
ENV PATH="${ELASTIX_DIR}/bin:${PATH}"
ENV LD_LIBRARY_PATH="${ELASTIX_DIR}/lib"

# Install TexLive with basic scheme.
ARG SCHEME="basic"
ARG TEX_DIR="${HOME}/tex/latest"
RUN wget https://mirror.ctan.org/systems/texlive/tlnet/install-tl-unx.tar.gz \
    && tar -zxvf install-tl-unx.tar.gz \
    && ./install-tl*/install-tl --scheme ${SCHEME} --texdir ${TEX_DIR} --no-interaction\
    && rm -rf install-tl* \

# Add TexLive executables directory to path,
# and install packages needed to use LaTeX with Matplotlib.
USER root
ENV PATH="${TEX_DIR}/bin/x86_64-linux:${PATH}"
RUN tlmgr install cm-super collection-fontsrecommended dvipng type1cm underscore
USER ${NB_UID}

# Enable jupyter lab
ENV JUPYTER_ENABLE_LAB="yes"

WORKDIR "${HOME}/workdir"

LABEL "org.opencontainers.image.description"="Toolkit for radiotherapy image analysis"
LABEL "org.opencontainers.image.documentation"="https://scikit-rt.github.io/scikit-rt/"
LABEL "org.opencontainers.image.source"="https://github.com/scikit-rt/scikit-rt"
