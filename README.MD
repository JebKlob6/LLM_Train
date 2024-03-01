# Project Overview

This project involves using machine learning models to process and analyze data from a inputs or a file. It leverages a series of models to generate responses based on the context and questions provided.

The project utilizes the following models for analysis:

- `mistralai/Mistral-7B-Instruct-v0.2`
- `MBZUAI/LaMini-Flan-T5-248M`
- `philschmid/bart-large-cnn-samsum`
- `google/flan-t5-base`

These models are used to interpret the context and questions, generating relevant responses based on the input data.

## Download MiniConda
```
    - https://docs.anaconda.com/free/miniconda/
```

## Environment Setup

1. **Ensure Conda is Installed**: Make sure you have Conda installed on your system. You can install either Miniconda or Anaconda to get started. Once installed verify by typing conda into termina


2. **Create the Environment**: Navigate to the directory via terminal containing the `environment.yml` file in your terminal and run the following command:

    ```
    conda env create -f environment.yml
    ```

    This command will create a new Conda environment with all the necessary dependencies specified in the `environment.yml` file.

3. **Activate the Environment**: Once the environment is created, you can activate it by running:

    ```
    conda activate mlx-2
    ```

    Run in Terminal

4. **Verify Installation**: To ensure that the environment was set up correctly, you can list the installed packages using:

    ```
    conda list
    ```


## Importing Project into PyCharm

To work on this project within PyCharm, follow these steps to import the project and set up the Python interpreter:

1. **Open PyCharm**: Start PyCharm and select `Open` on the welcome screen or `File > Open...` from the menu if you already have a project open.

2. **Select Project Directory**: Navigate to the directory where your project is located, select it, and click `OK`. PyCharm will open the project.

3. **Set Python Interpreter**: To set the Python interpreter to the one in your newly created Conda environment, follow these steps:
   
   - Go to and click `(MenuBar)PyCharm > Settings` (on macOS).
   - Navigate to `Project: DA > Python Interpreter`.
   - Click on the gear icon next to the Python interpreter path, and select `Add...`.
   - In the Add Python Interpreter dialog, select `Conda Environment` on the left.
   - Choose `Existing environment`, then click on the three dots (Down Carrot) to browse and select the Python interpreter from your newly created Conda environment. It is usually located in your Anaconda/Miniconda directory under `envs/mlx/bin/python`.
   - Click `OK` to apply the changes.

4. **Verify Interpreter Setup**: After setting the Python interpreter, you can verify the setup by running a simple script to ensure that the interpreter is correctly configured and can import the project's dependencies without issues.

