# Neural Network Learning Project

This project is designed for learning and experimenting with neural networks using Python and Jupyter Notebooks.

## Setup Instructions

### 1. Clone the Repository
Clone this repository to your local machine.

### 2. Install Poetry
If you don't have Poetry installed, follow the instructions at [https://python-poetry.org/docs/#installation](https://python-poetry.org/docs/#installation).

### 3. Install Dependencies
Open a terminal in the project directory and run:
```
poetry install
```
This will create a virtual environment and install all required packages.

### 4. Add Jupyter Support
If not already installed, add Jupyter and the IPython kernel:
```
poetry add notebook ipykernel
```

### 5. Activate the Virtual Environment
Run:
```
poetry shell
```

### 6. Launch JupyterLab
```
poetry run jupyter lab
```
This will start the server and open JupyterLab in your browser at `http://localhost:8888`.

### 7. Stop JupyterLab
Press `Ctrl + C` in the terminal where the server is running, then confirm with `y`.

### 8. (Alternative) Launch in VS Code
- Open VS Code in this project folder.
- Create or open a `.ipynb` notebook file.
- Select the Python interpreter from your Poetry environment as the kernel.

## Useful Jupyter Shortcuts in VS Code
- Run cell: `Shift + Enter`
- Insert cell below: `B`
- Insert cell above: `A`
- Delete cell: `DD` (press D twice)
- Convert to code cell: `Y`
- Convert to markdown cell: `M`
- Move cell up: `K`
- Move cell down: `J`

## Acknowledgments
- Thanks to Andrej Karpathy for his excellent course on neural networks.
- Thanks to the open-source community for their contributions to the tools and libraries used in this project.