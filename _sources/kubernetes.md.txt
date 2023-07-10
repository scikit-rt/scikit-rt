# Scikit-rt with Kubernetes

The [scikit-rt Docker image](https://github.com/scikit-rt/scikit-rt/pkgs/container/scikit-rt) can be used with [JupyterHub](https://hub.jupyter.org/) on
a [Kubertetes](https://kubernetes.io/) cluster, to create a standardised
scikit-rt environment for a group of users, for example for training.

[Zero to JupyterHub with Kubernetes](https://z2jh.jupyter.org/) provides instructions for setting up a Kubernetes cluster on cloud
resources, and for installing and customising JupyterHub.  The instructions
below outline the procedure for a local Kubernetes cluster, mainly
for demonstration purposes.  The customisation relative to scikit-rt is
the same for cluster set up on cloud rresources. The instructions
have been tested on a MacBook, but should work also on other platforms.

## Software installation

The following are needed:

1. Install and start [Docker Desktop](https://docs.docker.com/desktop/).

2. Install the Kubernetes command line tool:
   [kubectl](https://kubernetes.io/docs/tasks/tools/).

3. Install [minikube](https://minikube.sigs.k8s.io/docs/) software
   for creating a local Kubernetes cluster, and start your
   cluster - steps 1 and 2 of:
   [minikube start](https://minikube.sigs.k8s.io/docs/start/).

4. Install a package manager for Kubernetes:
   [Helm][https://helm.sh/docs/intro/install/].

## JupyterHub configuration and start up

5. Update the list of repositories of Kubernetes configuration files
   (Helm charts) known to Helm, to include the
   [JupyterHub Helm chart repository](https://jupyterhub.github.io/helm-chart/):

   ```
   helm repo add jupyterhub https://jupyterhub.github.io/helm-chart/
   helm repo update
   ```

6. Download the scikit-rt customisation of the JupyterHub configuration:

   ```
   curl https://raw.githubusercontent.com/scikit-rt/scikit-rt/master/skrt-jupyterhub.yaml > skrt-jupyterhub.yaml
   ```

7. Install and start JupyterHub, with scikit-rt customisation, on your
   Kubernetes cluster:

   ```
   helm upgrade --cleanup-on-fail \
    --install jupyterhub jupyterhub/jupyterhub \
    --namespace jupyter \
    --create-namespace \
    --version=2.0.0 \
    --values skrt-jupyterhub.yaml
   ```

   This may take some time to complete, while Docker images are downloaded
   in the background.

## Connecting to JupyterHub

8. Check the status of pods in the `jupyter` namespace of
   your Kubernetes cluster:
   ```
   kubectl get pods -n jupyter
   ```

9. When all pods are shown as running, connect to JupyterHub
   with the command:

   ```
   minikube service -n jupyter proxy-public
   ```

   This will print the connection URL (the second URL shown), and
   will open it in a web browser.  Leave the command running.

10. As authentication hasn't been enabled for this demonstration,
    you can log in to JupyterHub with any combination of username
    and password.

## Working with scikit-rt on JupyterHub

11. The JupyterHub session runs from the container directory /home/jovyan.  You
    should have access to a directory `examples`, containing the notebooks
    from [Scikit-rt by examples]
    (https://scikit-rt.github.io/scikit-rt/examples.html).  The environment
    is set up so that you can import scikit-rt (`import skrt`) and its
    dependent packages, and so that you can perform image registration
    with [elastix](https://elastix.lumc.nl/) and
    [NiftyReg](http://cmictig.cs.ucl.ac.uk/wiki/index.php/NiftyReg).

12. To save to the local file system a notebook or other file from
    your JupyterHub session, right-click on the file in the Jupyter
    `File Browser`, and select `Download`.  If the `File Browser`
    isn't visible, click on the folder icon towards the top left
    of the Jupyter part of your web browser. 

## Disconnecting and reconnecting

13. To disconnect form JupyterHub, select `File` from
    the Jupyter part of your web browser, and then select `Log Out`.

14. Optionally disable further connections by interupting the
    command given in step 9:

    ```
    minikube service -n jupyter proxy-public
    ```

    Reissuing the command will reenable connections.

15. Reconnect to JupyterHub by logging in again at the URL for
    the `proxy-public` service (step 9 or step 14).  JupyterHub
    provides a multi-workspace environment.  If you use the same
    username as previously, you will find the work environment,
    and any file modifications, the same as when you logged out.
    If you use a different username, you will find a reinitialised
    environment.  Each username with which you connect will have
    an independent workspace.  (This functionality becomes more
    useful when supporting a group of users, with authenticated
    login.)

## Stopping and restarting Kubernetes cluster

16. To stop your Kubernetes cluster, use:

    ```
    minikube stop
    ```

17. To restart your Kubernetes cluster, use:

    ```
    minikube start
    ```

18. To reconnect to JupyterHub after restarting your Kubernetes cluster,
    repeat steps 9 and 15.

## Deleting JupyterHub and Kubernetes cluster

19. Before deleting JupyterHub or your Kubernetes cluster, ensure
    that you have copies outside of the containers used of any files
    that you want to keep.  For JupyterHub, this can be achieved, for
    example, by downloading files to the local file system, as
    outlined in step 12.

20. To delete the Jupyter resources on your Kubernetes cluster, without
    deleting the cluster, use:

    ```
    kubectl delete all --all -n jupyter
    kubectl delete namespace jupyter
    ```

21. To delete your Kubernetes cluster, use:

    ```
    minikube delete
    ```
