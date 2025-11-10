# Pokémon Yellow AI

This project uses Hierarchical Reinforcement Learning (HRL) to train an AI to play Pokémon Yellow. The AI is structured in a three-tiered hierarchy: Specialists, a Manager, and a Meta-Manager.

## Features

- **Hierarchical Reinforcement Learning (HRL):** A multi-layered AI architecture for complex decision-making.
- **Specialist Models:** Individual models trained for specific tasks like exploration, battling, healing, and shopping.
- **Manager AI:** A mid-level AI that directs the Specialists based on the current game state.
- **Meta-Manager AI:** A top-level AI that sets long-term goals for the Manager.
- **Training & Watching Modes:** You can either train the AI models from scratch or watch the trained models play the game.
- **Debugging Tools:** Includes tools for memory scanning, watching memory values, and debugging text hashes.

## Prerequisites

- Python 3.x
- A legitimate copy of the Pokémon Yellow ROM file, named `PokemonYellow.gb`.

## Setup

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd "Pokemon Yellow AI"
    ```

2.  **Place the ROM:**
    Put your `PokemonYellow.gb` file in the root directory of the project.

3.  **Install Dependencies:**
    Run the `menu.bat` script and select option `8` to install the required Python packages from `requirements.txt`. Alternatively, you can run this command in your terminal:
    ```bash
    pip install -r requirements.txt
    ```

4.  **Create Initial Save State:**
    Run `menu.bat` and select option `10` to run the `create_save_state.py` script. This will guide you through creating the necessary initial save file for the AI to start from. You will also need to create save states for different scenarios like battles, healing, etc. and place them in the `states` directory.

## Usage

The primary way to interact with this project is through `menu.bat`.

### Training the AI

The AI is trained in a bottom-up fashion. You must train the models in this order:

1.  **(L1) Train Specialists:**
    -   In `menu.bat`, select option `1`.
    -   You can then choose which specialist to train (e.g., Exploration, Battle, Healer).
    -   Training requires corresponding save states in the `states` directory (e.g., `battle_00.state`, `healer_01.state`).

2.  **(L2) Train the Manager:**
    -   After training all the specialists, select option `2` in `menu.bat`.
    -   This trains the Manager AI, which learns to control the specialists.

3.  **(L3) Train the Meta-Manager:**
    -   Once the Manager is trained, select option `3` in `menu.bat`.
    -   This trains the top-level "CEO" AI that sets goals for the Manager. This process is very slow.

### Watching the AI

You can watch the AI play at any level of the hierarchy.

-   **(L1) Watch a Specialist:** Select option `4` to watch a single specialist model perform its task.
-   **(L2) Watch the Manager:** Select option `5` to watch the Manager AI control the specialists without any high-level goals.
-   **(L3) Watch the Meta-Manager:** Select option `6` to watch the full AI stack in action.
-   **Watch the Rules-Based Manager:** Select option `7` to watch a non-learning, rules-based version of the manager.

### TensorBoard

To monitor the training progress, run `menu.bat` and select option `9`. This will start TensorBoard, and you can view the logs at `http://localhost:6006`.

## Project Structure

-   `PokemonYellow.gb`: The game ROM (must be provided by the user).
-   `menu.bat`: The main script for interacting with the project.
-   `requirements.txt`: A list of Python dependencies.
-   `src/`: Contains the source code for the AI agents, environment, and utilities.
    -   `agents/`: The policy networks for the AI.
    -   `env/`: The custom Gymnasium environment for Pokémon Yellow.
    -   `utils/`: Utility scripts for callbacks, memory, etc.
-   `models/`: Stores the trained AI models.
-   `logs/`: Contains TensorBoard logs for monitoring training.
-   `states/`: Stores game save states (`.state` files) used for training specific scenarios.
-   `train_*.py`: Scripts for training the different levels of the AI.
-   `watch_*.py`: Scripts for watching the trained AI models play.
-   `debug_*.py`: Various utility scripts for debugging.

## How It Works

The AI uses a Hierarchical Reinforcement Learning (HRL) architecture with three levels:

1.  **Specialists (L1):** These are low-level agents trained to perform specific, short-term tasks. Examples include:
    -   `exploration_model`: Navigates the game world.
    -   `battle_model`: Fights Pokémon.
    -   `healer_model`: Navigates to a Pokémon Center to heal.
    -   And more for shopping, managing inventory, and switching Pokémon.

2.  **Manager (L2):** This is a mid-level agent that does not directly control the game. Instead, it decides which Specialist to activate based on the current game context. For example, if a wild Pokémon appears, the Manager will activate the `battle_model`.

3.  **Meta-Manager (L3):** This is the highest-level agent, acting as the "CEO" of the AI. It sets long-term goals for the Manager to achieve, such as "get badge 1" or "level up Pokémon". The Manager then translates these goals into actions by deploying the appropriate Specialists.

This hierarchical structure allows the AI to break down the complex goal of playing Pokémon Yellow into smaller, more manageable tasks.

## Utilities & Debugging

The `menu.bat` provides access to several utilities:

-   **Metrics Exporter:** Export training metrics to a summary file.
-   **GUI Memory Scanner:** A tool to find memory addresses in the game.
-   **Memory Watcher:** Watch memory values change in real-time.
-   **Text Hash Debugger:** Capture the hash of on-screen text.

You can also run the training scripts in debug mode from the menu, which will run on a single CPU and render the game window.

## Contributing

Contributions are welcome! Please feel free to submit a pull request.

## License

This project is unlicensed.
