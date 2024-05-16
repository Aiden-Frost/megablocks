# Running Megablocks on NYU HPC

Below are the steps to run Megablocks on NYU HPC:

1. SSH into the NYU HPC burst node:
    ```bash
    ssh burst
    ```

2. Allocate resources for running Megablocks:
    ```bash
    srun --account=csci_ga_3033_077-2024sp --partition=n1s8-v100-1 --gres=gpu:v100:1 --time=02:00:00 --wait=300 --pty /bin/bash
    ```

3. Create a symbolic link for `ld.so.cache`:
    ```bash
    touch ld.so.cache~
    realpath ld.so.cache~
    ```

4. Clone the Megablocks repository:
    ```bash
    git clone --recursive https://github.com/Aiden-Frost/megablocks.git
    cd megablocks
    ```

5. Build the Nvidia pytorch Singularity container:
    ```bash
    singularity build megablocks.sif docker://nvcr.io/nvidia/pytorch:24.04-py3
    ```

6. Run the Megablocks Singularity container:
    ```bash
    singularity exec --nv --bind /home/<net_id>/tmp/ld.so.cache:/etc/ld.so.cache:ro /home/<net_id>/megablocks/megablocks.sif /bin/bash
    ```
   
7. Inside the Megablocks container, install the required dependencies:
    ```bash
    ldconfig /.singularity.d/libs
    pip install stanford-stk==0.0.6
    pip install flash-attn
    pip install .
    pip install grouped_gemm
    ```

Note: Replace `<net_id>` with your NYU net ID. If you are facing broken pipe issue when installing any packages, then 
install them via nohup command. Example: If (pip install .) is taking too long to install, then install via
```bash
(nohup pip install .) &
```
This will do the installation in background. Then you can hit `ctrl+c` and then run `top` and wait until the installation is done.
Which you can check using `jobs` command.
