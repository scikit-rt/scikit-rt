# Scikit-rt with Docker

A [Docker](https://www.docker.com/) image for
[scikit-rt](https://scikit-rt.github.io/scikit-rt/) is built for each release,
based on a [Dockerfile](../../Dockerfile), and is pushed to the
[GitHub container registry](https://docs.github.com/en/packages/working-with-a-github-packages-registry/working-with-the-container-registry).
The image includes the image-registration packages
[elastix](https://elastix.lumc.nl/) and
[NiftyReg](http://cmictig.cs.ucl.ac.uk/wiki/index.php/NiftyReg).  When run,
it starts a [JupyterLab](https://jupyterlab.readthedocs.io/en/latest/)
session, within the scikit-rt environment.

If you have an installation of [Docker](https://www.docker.com/),
one approach to working interactively sith scikit-rt is as follows:

1. Download the scikit-rt Docker image:

   ```
   docker pull ghcr.io/scikit-rt/scikit-rt:latest
   ```

   The `latest` tag can optionally be replaced by a specific tag for any of the
   [tagged image versions](https://github.com/scikit-rt/scikit-rt/pkgs/container/scikit-rt).

2. Choose a work directory on your local file system, copy here any files
   and data that you would want to be able to access with scikit-rt, then
   start a Docker container:

   ```
   docker run -v /path/to/work/directory:/home/jovyan/work -p 8888:8888 ghcr.io/scikit-rt/scikit-rt
   ```

   The following are noted:

   - The argument to `-v` maps a work directory on the local file system
     (`/path/to/work/directory`) to the work directory of the docker container
     (`/home/jovyan/work`).  The former can be any existing path to which
     you have write access; the latter should be a subdirectory of
     `/home/jovyan'.  Paths should always be absolute (not relative).
   - The argument to `-p` maps the server port (first value: 8888) on the local
     machine to the server port (second value: 8888) on the container side.
     The port number on the local machine should be different from
     any other port numbers used by applications running locally; the
     port number on the container side is fixed.

3. Information will be printed to screen about Jupyter start-up.  Copy the
   last URL listed (typically starting: http://127.0.0.1:8888), and open this
   URL in a web browser.  This should open a jupyter lab session.
   If the session fails to open:

   - Check that you've correctly copied the URL, including all characters 
     of the access token.
   - Check that the local port number used in step 2 isn't the same as the
     port number used by another application running locally.

   Keep a record of the URL for the Jupyter session, as this will be needed
   if you end the session and subseqently want to resume it.

4. The Jupyter session runs in the container from the directory
   `/home/jovyan`.  You should have access to a directory `examples`,
   containing the notebooks from [Scikit-rt by examples](examples.md).
   You should also have access to the work directory shared with your
   local file system, and specified in step 2.  The environment is set up
   so that you can import scikit-rt (`import skrt`) and its dependent
   packages, and so that you can perform image registration with 
   [elastix](https://elastix.lumc.nl/) and
   [NiftyReg](http://cmictig.cs.ucl.ac.uk/wiki/index.php/NiftyReg).

5. Before ending a session or deleting the container, ensure that all files
   that you want to keep have been copied to the work directory specified
   in step 2, and are showing up on your local file system.  Files in
   other directories are preserved between sessions, but are lost when
   the container is deleted.

6. To end your session, and stop the container, select `File` from
   the Jupyter part of your web browser, and then select
   `Shut Down`.

6. To resume a session:

   - Determine the identifier, `<container_id>`, of your container:

     ```
     docker ps -a
     ```

   - Restart the container:     

     ```
     docker container restart <container_id>
     ```

   - Open in a web browser the URL recorded in step 3.  In case you have
     no record of the URL, you should be able to recover it from the
     container log, which can be accessed with:

     ```
     docker logs <container_id>
     ```

   **Warning**: If you use `docker run`, as in step 2, to start the container,
   rather than `docker container restart`, this will create a new
   container instance.  Modifications made outside the work directory
   of the previous container instance won't be visible.  The previous
   instance remains available, and may be restarted as outlined in this step.

7. To delete a container:

   - Make sure that you have a copy outside the container
     of all files that you want to keep, for example by copying them to 
     the work directory shared between container and local file system,
     and specified in step 2.

   - Determine the identifier, `<container_id>`, and status of the
     container to be deleted:

     ```
     docker ps -a
     ```

   - If the container is running, then stop it.

     ```
     docker stop <container_id>
     ```

     This command can also be run if in doubt - if the container is
     already stopped, it will have no effect.

   - Remove the stopped container:

     ```
     docker rm <container_id>
     ```
