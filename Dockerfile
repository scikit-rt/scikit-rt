FROM jupyter/minimal-notebook

# Install specific version of python.
RUN conda install python=3.10

# Install scikit-rt.
RUN git clone https://github.com/scikit-rt/scikit-rt \
    && python -m pip install -e scikit-rt \
    && mkdir ${HOME}/workdir

# Copy example Jupyter notebooks.
ARG NOTEBOOKS="examples/notebooks"
ARG WORKDIR="${HOME}/workdir"
COPY ${NOTEBOOKS}/patient_datasets.ipynb ${WORKDIR}
COPY ${NOTEBOOKS}/plotting_demo.ipynb ${WORKDIR}
COPY ${NOTEBOOKS}/application_demo.ipynb ${WORKDIR}
COPY ${NOTEBOOKS}/image_processing.ipynb ${WORKDIR}
COPY ${NOTEBOOKS}/roi_intensities.ipynb ${WORKDIR}
COPY ${NOTEBOOKS}/dose_volume_rois.ipynb ${WORKDIR}
COPY ${NOTEBOOKS}/eqd.ipynb ${WORKDIR}
COPY ${NOTEBOOKS}/synthetic_dicom_dataset.ipynb ${WORKDIR}
COPY ${NOTEBOOKS}/grid_creation.ipynb ${WORKDIR}
COPY ${NOTEBOOKS}/image_registration_checks.ipynb ${WORKDIR}

# Install elastix.
ARG ELASTIX_VERSION="5.1.0"
ARG ELASTIX_LINUX="elastix-${ELASTIX_VERSION}-linux"
ARG ELASTIX_ZIP="${ELASTIX_LINUX}.zip"
RUN wget https://github.com/SuperElastix/elastix/releases/download/${ELASTIX_VERSION}/${ELASTIX_ZIP} \
    && unzip ${ELASTIX_ZIP} -d ${ELASTIX_LINUX} \
    && rm ${ELASTIX_ZIP} \
    && chmod a+x ${ELASTIX_LINUX}/bin/elastix \
    && chmod a+x ${ELASTIX_LINUX}/bin/transformix

# Set up environment for running elastix.
ARG ELASTIX_DIR="${HOME}/${ELASTIX_LINUX}"
ENV PATH="${ELASTIX_DIR}/bin:${PATH}"
ENV LD_LIBRARY_PATH="${ELASTIX_DIR}/lib"

# Install TexLive with basic scheme.
ARG SCHEME="basic"
ARG TEX_DIR="${HOME}/tex/latest"
RUN wget https://mirror.ctan.org/systems/texlive/tlnet/install-tl-unx.tar.gz \
    && tar -zxvf install-tl-unx.tar.gz \
    && ./install-tl*/install-tl --scheme ${SCHEME} --texdir ${TEX_DIR} --no-interaction \
    && rm -rf install-tl*

USER root

# Add TexLive executables directory to path,
# and install packages needed to use LaTeX with Matplotlib.
ENV PATH="${TEX_DIR}/bin/x86_64-linux:${PATH}"
RUN tlmgr install cm-super collection-fontsrecommended dvipng type1cm underscore

# Install library needed by elastix
RUN apt-get update \
    && apt-get -y install libgomp1

USER ${NB_UID}

# Enable jupyter lab
ENV JUPYTER_ENABLE_LAB="yes"

WORKDIR "${HOME}/workdir"

LABEL "org.opencontainers.image.description"="Toolkit for radiotherapy image analysis"
LABEL "org.opencontainers.image.documentation"="https://scikit-rt.github.io/scikit-rt/"
LABEL "org.opencontainers.image.source"="https://github.com/scikit-rt/scikit-rt"
