# How to use the cluster

## Login
Activate the University of Osnabrueck's VPN and make sure no other VPNs are active, then run:
```bash
ssh <username>@hpc3.rz.uos.de
```
If no error occurs, this will prompt you to enter your password. You are not going to see the input, so just enter it and hit `ENTER`.

## Interactive session

To allocate a compute node and get a shell prompt run:

```bash
srun --pty /bin/bash
```

To end the session type
```bash
exit
```
and hit `ENTER`.


## Slurm

For detailed information on a slurm command you can run
```bash
man <command_name>
```

### Submit a job

To submit a job using `slurm`, create a script specifying the job. Here's an example. Replace `job_name`, `HH:MM:SS`, `XXgb`, `X`, `partition_name`, `/path/to/miniconda3`, `environment_name`, `/path/to/project/directory`, `script.py`, and the arguments with the appropriate values for your job.
```sh
#!/bin/bash
#SBATCH -J job_name
#SBATCH --time=HH:MM:SS
#SBATCH --mem=XXgb
#SBATCH --gpus=X
#SBATCH --ntasks=X
#SBATCH --cpus-per-task=X
#SBATCH --partition=partition_name
#SBATCH --gpu-bind=single:1

# Source the .bashrc file
source $HOME/.bashrc

# Add the Conda installation to the PATH
export PATH="/path/to/miniconda3/bin:$PATH"

# Activate the Conda environment
source /path/to/miniconda3/etc/profile.d/conda.sh
conda activate environment_name

# Navigate to the project directory
cd /path/to/project/directory

# Run the Python script using srun
srun python script.py --arg1 value1 --arg2 value2

# Optional: Add a sleep command to allow time for saving output files
srun sleep 10
```

With this script, to schedule your job, run:
```bash
sbatch <script_file_path>
```
For example:
```bash
sbatch gridsearch.sh
```

This will return a reference to your job:

```bash
Submitted batch job 6012345
```

The number at the end (6012345) is the `JOBID` assigned by Slurm.

### View job queue
```bash
squeue
```

This command will display the list of queued and running jobs. The output of squeue includes the following columns:

- JOBID: The unique identifier assigned to each job.
- PARTITION: The partition or queue to which the job is assigned.
- NAME: The name of the job.
- USER: The username of the user who submitted the job.
- ST: The state of the job. Common states include:

  - PD: Pending (waiting for resources)
  - R: Running
  - CG: Completing
  - CD: Completed
- TIME: The time the job has been running or waiting in the queue.
- NODES: The number of nodes allocated to the job.
- NODELIST(REASON): The list of nodes assigned to the job or the reason why the job is waiting (e.g., Resources, Priority).

#### filter squeue output
The queue can be pretty long. To get a more concise overview over your jobs you can filter the output of squeue based on
- job states: use the -t or --states option followed by the desired state(s)
- user name: use the -u or --user option followed by your username

Here's a few examples

##### view only your jobs
```bash
squeue -u <your_username>
```

##### view only your running jobs
```bash
squeue -u <your_username> -t R
```

##### view only your pending and running jobs
```bash
squeue -u your_username -t PD,R
```

By default, squeue displays a set of default columns. However, you can customize the output format using the `-o` or `--format` option to include specific information that you're interested in.

For example, to display the job ID, job name, state, and time used:

```bash
squeue -u <your_username> -o "%.10i %.30j %.2t %.10M"
```

Here's what each format specifier represents:

- `%.10i`: Job ID, right-justified, 10 characters wide
- `%.30j`: Job name, left-justified, 30 characters wide
- `%.2t`: Job state, right-justified, 2 characters wide
- `%.10M`: Time used by the job, right-justified, 10 characters wide

By default, `squeue` provides a snapshot of the job queue at the moment you run the command. To get real-time updates, you can use the `-i` or `--iterate` option followed by the update interval in seconds.

For example, to update the queue information every 5 seconds
When the job runs, a file named `slurm-<jobid>.out` will be created in the directory you started the job from. This file contains the job's output that you would usually see printed to the terminal. You can look at the content by running:

```bash
squeue -u your_username -i 5
```

This command will continuously update the output every 5 seconds until you stop it with `Ctrl+C`.

```bash
cat slurm-<jobid>.out
```
Replace `<jobid>` with your actual `JOBID` assigned by Slurm.

### Cancel a job
```bash
scancel <jobid>
```

## Dealing with conda environments

### List existing environments

This command will list all the discoverable Conda environments.

```bash
conda info --envs
```

If the desired environment is not listed, you may need to modify the PATH to include the Conda installation directory. Run the following command, replacing `/path/to/miniconda3` with the actual path to your Miniconda installation

### Set correct environment path
```bash
export PATH="/path/to/miniconda3/bin:$PATH"
```

### Veryfiy environment
You can veryfiy that the correct conda environment is running by running:

```bash
which conda
```

It should display the path to the Conda executable in your Miniconda installation directory.

### Source the Conda initialization script 

Source the Conda initialization script by running
```bash
source /path/to/miniconda3/etc/profile.d/conda.sh
```
Replace `/path/to/miniconda3` with the actual path to your Miniconda installation obtained by `conda info --envs`. 

### Activate conda environment

```bash
conda activate <environment_name>
```

Verify that the environment is activated by checking the command prompt. It should display the name of the active environment in parentheses, like this:

```bash
(environment_name) user@cluster:~$
```

## Issues and Troubleshooting

### Conda command not found

If you encounter the error message `conda: command not found`, it means that the Conda executable is not in your system's `PATH`. To resolve this issue:

1. Locate the Conda installation directory on your cluster. It is usually in your home directory or a shared directory.
2. Add the Conda installation directory to your `PATH` by running
   ```bash
   export PATH="/path/to/miniconda3/bin:$PATH"
   ```
3. Verify that the Conda command is now accessible
    ```bash
    which conda
    ```

### Conda environment not found

If you receive the error message "Could not find conda environment: environment_name", it indicates that the specified Conda environment does not exist or is not discoverable. To resolve this issue:
1. Check the list of available Conda environments
   ```bash
   conda info --envs
   ```
2. If the desired environment is not listed, make sure you have activated the correct Conda installation and sourced the initialization script
    ```bash
    export PATH="/path/to/miniconda3/bin:$PATH"
    source /path/to/miniconda3/etc/profile.d/conda.sh
    ```


### Module not found error
If you encounter the error message "ModuleNotFoundError: No module named 'your_module'," it means that the specified Python module is not installed in the active Conda environment. To resolve this issue:
1. activate your conda environment
2. install the missing module (using conda or pip)
    ```bash
    conda install <module_name>
    ```
    or
    ```bash
    pip install <module_name>
    ```
3. verify that the module is installed by running:
    ```bash
    python -c "import <module_name>"
    ```
    Replace `<module_name>` by your concrete module's name.
    If this passes without an error, the module is correctly installed.
