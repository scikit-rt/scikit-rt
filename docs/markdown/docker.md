# Scikit-rt with Docker

A [Docker](https://www.docker.com/) image for
[scikit-rt](https://scikit-rt.github.io/scikit-rt/) is built for each release
(see; <a href="https://github.com/scikit-rt/scikit-rt/blob/master/Dockerfile" type="text/plain">Dockerfile]</a>), and
 is pushed to the
[GitHub container registry](https://docs.github.com/en/packages/working-with-a-github-packages-registry/working-with-the-container-registry).
When the image is run, it starts a
[JupyterLab](https://jupyterlab.readthedocs.io/en/latest/)
session within the scikit-rt environment, with the image-registration packages
[elastix](https://elastix.lumc.nl/) and
[NiftyReg](http://cmictig.cs.ucl.ac.uk/wiki/index.php/NiftyReg) available.

If you have an installation of [Docker](https://www.docker.com/),
one approach to working interactively with scikit-rt is as outlined below.

## Running scikit-rt Docker image

1. Download the scikit-rt Docker image:

   ```
   docker pull ghcr.io/scikit-rt/scikit-rt:latest
   ```

   The `latest` tag can optionally be replaced by a specific tag for any of the
   [tagged image versions](https://github.com/scikit-rt/scikit-rt/pkgs/container/scikit-rt).

2. Choose a work directory on your local file system, copy here any files
   and data that you would want to be able to access with scikit-rt, then
   run the scikit-rt Docker image in a container:

   ```
   docker run -v /path/to/work/directory:/home/jovyan/work -p 9090:8888 ghcr.io/scikit-rt/scikit-rt
   ```

   The following are noted:

   - The argument to `-v` maps a work directory on the local file system
     (`/path/to/work/directory`) to the work directory of the docker container
     (`/home/jovyan/work`).  The former can be any existing path to which
     you have write access; the latter should be a subdirectory of
     `/home/jovyan`.  Paths should always be absolute (not relative).
   - The argument to `-p` maps the server port (9090) on the local
     machine to the server port (8888) on the container side.
     The port number on the local machine can be any port number not
     used by an application running locally; the port number on the
     container side is fixed.

3. Information will be printed to screen about Jupyter start-up.  Copy the
   last URL listed (starting: http://127.0.0.1:8888) to the address bar
   of a web browser.  Replace the container port number (8888) by the local
   port number chosen in step 2 (for example, 9090), and open the resulting
   URL.  This should give access to a jupyter lab session.  If the session
   fails to open:

   - Check that you have the correct URL, including all characters 
     of the access token.
   - Check that the local port number used in step 2 isn't the same as the
     port number used by another application running locally.

   Keep a record of the URL for the Jupyter session, as this will be needed
   if you end the session, and subseqently want to resume it.

4. The Jupyter session runs in the container from the directory
   `/home/jovyan`.  You should have access to a directory `examples`,
   containing the notebooks from [Scikit-rt by examples](examples.md).
   You should also have access to the work directory shared with your
   local file system, and specified in step 2.  The environment is set up
   so that you can import scikit-rt (`import skrt`) and its dependent
   packages, and so that you can perform image registration with 
   [elastix](https://elastix.lumc.nl/) and
   [NiftyReg](http://cmictig.cs.ucl.ac.uk/wiki/index.php/NiftyReg).

## Ending a session

5. Before ending a session or deleting the container, ensure that all files
   that you want to keep have been copied to the work directory specified
   in step 2, and are showing up on your local file system.  Files in
   other directories are preserved between sessions, but are lost when
   the container is deleted.

6. To end your session, and stop the container, select `File` from
   the Jupyter part of your web browser, and then select
   `Shut Down`.

## Resuming a session

7. To resume a session:

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

## Connecting to a container

8. To connect to a (running) container:

   - Determine the identifier, `<container_id>`, of your container:

     ```
     docker ps -a
     ```

   - To connect to a bash shell in the container:

     - without root privileges:
     ```
     docker exec -it <container-id> bash
     ```

     - with root privileges:
     ```
     docker exec -u root -it <container-id> bash
     ```

9. To terminate the connection, from the bash shell type `exit`.

## Deleting a container

10. To delete a container:

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

    - Remove the stopped container:

      ```
      docker rm <container_id>
      ```
