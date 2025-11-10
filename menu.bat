@echo off
title Pokemon AI Project Menu (Full HRL Stack)
set STATE_DIR=states

:menu
cls
echo ===================================
echo  Pokemon Yellow AI Project (Full HRL Stack)
echo ===================================
echo.
echo --- TRAINING (BOTTOM-UP) ---
echo 1. (L1) Train SPECIALISTS (explore, battle, healer, etc.)
echo 2. (L2) Train MANAGER (Controls Specialists)
echo 3. (L3) Train META-MANAGER (CEO - Sets Goals)
echo.
echo --- WATCHING ---
echo 4. (L1) Watch a single SPECIALIST
echo 5. (L2) Watch the trained MANAGER (No goals, just reacts)
echo 6. (L3) Watch the full META-MANAGER AI play (CEO)
echo 7. Watch the RULES-BASED Manager
echo.
echo --- UTILITIES ---
echo 8.  Install/Update Dependencies (pip install -r requirements.txt)
echo 9.  Start TensorBoard (View training progress)
echo 10. Run Initial Save State Creator (create_save_state.py)
echo 11. Run Metrics Exporter (export_metrics.py)
echo 12. Run Text Hash Debugger (debug_text_hash.py)
echo 13. (NEW) Run GUI Memory SCANNER (debug_memory_scanner.py)
echo 14. (OLD 13) Run Memory Watcher (debug_memory_watch.py)
echo.
echo --- DEBUGGING (1 CPU, VISIBLE) ---
echo 15. (L1) DEBUG a SPECIALIST
echo 16. (L2) DEBUG the MANAGER
echo 17. (L3) DEBUG the META-MANAGER (CEO)
echo.
echo 18. Exit
echo.
set /p "choice=Enter your choice (1-18): "

if "%choice%"=="1" goto train_specialist
if "%choice%"=="2" goto train_manager
if "%choice%"=="3" goto train_meta
if "%choice%"=="4" goto watch_specialist
if "%choice%"=="5" goto watch_manager
if "%choice%"=="6" goto watch_meta
if "%choice%"=="7" goto watch_rules
if "%choice%"=="8" goto install
if "%choice%"=="9" goto tensorboard
if "%choice%"=="10" goto starter
if "%choice%"=="11" goto export_metrics
if "%choice%"=="12" goto debug_hash
if "%choice%"=="13" goto debug_scanner
if "%choice%"=="14" goto debug_memory
if "%choice%"=="15" goto debug_specialist
if "%choice%"=="16" goto debug_manager
if "%choice%"=="17" goto debug_meta
if "%choice%"=="18" goto exit

echo Invalid choice.
echo Press any key to return to the menu.
pause > nul
goto menu

:install
echo Installing dependencies from requirements.txt...
pip install -r requirements.txt
pip install keyboard
echo.
echo Dependencies installed. Press any key to return to the menu.
pause > nul
goto menu

:train_specialist
cls
echo ===================================
echo  (L1) Select a Specialist Lesson to Train
echo ===================================
echo.
echo --- CORE SPECIALISTS ---
echo.
echo 1. Lesson 1: EXPLORATION (exploration_model.zip)
echo.
echo    (Note: The lessons below can use multiple files like 'battle_00.state', 
echo    'battle_01.state', 'healer_00.state', 'healer_01.state', 'shopping_00.state', 
echo    'shopping_01.state, inventory_00.state', 'switch_00.state', etc.)
echo.
echo 2. Lesson 2: BATTLE (battle_model.zip)
echo.
echo --- GALAXY BRAIN SPECIALISTS ---
echo.
echo 3. Lesson 3: HEALER (healer_model.zip)
echo 4. Lesson 4: SHOPPING (shopping_model.zip)
echo 5. Lesson 5: INVENTORY (inventory_model.zip)
echo 6. Lesson 6: SWITCH (switch_model.zip)
echo.
echo 7. Return to Main Menu
echo.
set /p "lesson_choice=Enter your choice (1-7): "

if "%lesson_choice%"=="1" goto train_explore
if "%lesson_choice%"=="2" goto train_battle
if "%lesson_choice%"=="3" goto train_healer
if "%lesson_choice%"=="4" goto train_shopping
if "%lesson_choice%"=="5" goto train_inventory
if "%lesson_choice%"=="6" goto train_switch
if "%lesson_choice%"=="7" goto menu

echo Invalid choice.
echo Press any key to return.
pause > nul
goto train_specialist

:train_explore
echo Starting AI Training (Lesson: EXPLORATION)...
echo.
set TF_ENABLE_ONEDNN_OPTS=0
python train_specialist.py --lesson exploration --model_name exploration_model
echo.
echo Training finished or was interrupted.
echo Press any key to return to the menu.
pause > nul
goto menu

:train_battle
echo Starting AI Training (Lesson: BATTLE)...
IF NOT EXIST %STATE_DIR%\battle.state ( IF NOT EXIST %STATE_DIR%\battle_*.state (
    echo.
    echo ERROR: No battle save states found.
    echo Please create 'battle.state' OR 'battle_00.state', 'battle_01.state', etc.
    echo.
    pause
    goto menu
))
echo Battle save state(s) found.
echo Starting training...
echo.
set TF_ENABLE_ONEDNN_OPTS=0
python train_specialist.py --lesson battle --model_name battle_model
echo.
echo Training finished or was interrupted.
echo Press any key to return to the menu.
pause > nul
goto menu

:train_healer
echo Starting AI Training (Lesson: HEALER)...
IF NOT EXIST %STATE_DIR%\healer.state ( IF NOT EXIST %STATE_DIR%\healer_*.state (
    echo.
    echo ERROR: No healer save states found.
    echo Please create 'healer.state' OR 'healer_00.state', etc.
    echo.
    pause
    goto menu
))
echo Healer save state(s) found.
echo Starting training...
echo.
set TF_ENABLE_ONEDNN_OPTS=0
python train_specialist.py --lesson healer --model_name healer_model
echo.
echo Training finished or was interrupted.
echo Press any key to return to the menu.
pause > nul
goto menu

:train_shopping
echo Starting AI Training (Lesson: SHOPPING)...
IF NOT EXIST %STATE_DIR%\shopping.state ( IF NOT EXIST %STATE_DIR%\shopping_*.state (
    echo.
    echo ERROR: No shopping save states found.
    echo Please create 'shopping.state' OR 'shopping_00.state', etc.
    echo.
    pause
    goto menu
))
echo Shopping save state(s) found.
echo Starting training...
echo.
set TF_ENABLE_ONEDNN_OPTS=0
python train_specialist.py --lesson shopping --model_name shopping_model
echo.
echo Training finished or was interrupted.
echo Press any key to return to the menu.
pause > nul
goto menu

:train_inventory
echo Starting AI Training (Lesson: INVENTORY)...
IF NOT EXIST %STATE_DIR%\inventory.state ( IF NOT EXIST %STATE_DIR%\inventory_*.state (
    echo.
    echo ERROR: No inventory save states found.
    echo Please create 'inventory.state' OR 'inventory_00.state', etc.
    echo.
    pause
    goto menu
))
echo Inventory save state(s) found.
echo Starting training...
echo.
set TF_ENABLE_ONEDNN_OPTS=0
python train_specialist.py --lesson inventory --model_name inventory_model
echo.
echo Training finished or was interrupted.
echo Press any key to return to the menu.
pause > nul
goto menu

:train_switch
echo Starting AI Training (Lesson: SWITCH)...
IF NOT EXIST %STATE_DIR%\switch.state ( IF NOT EXIST %STATE_DIR%\switch_*.state (
    echo.
    echo ERROR: No switch save states found.
    echo Please create 'switch.state' OR 'switch_00.state', etc.
    echo.
    pause
    goto menu
))
echo Switch save state(s) found.
echo Starting training...
echo.
set TF_ENABLE_ONEDNN_OPTS=0
python train_specialist.py --lesson switch --model_name switch_model
echo.
echo Training finished or was interrupted.
echo Press any key to return to the menu.
pause > nul
goto menu

:train_manager
cls
echo ===================================
echo  (L2) Starting AI Training (MANAGER AI)
echo ===================================
echo.
echo This will train the high-level brain that CONTROLS the specialists.
echo This requires ALL specialist models (explore, battle, etc.) to exist first!
echo.
set TF_ENABLE_ONEDNN_OPTS=0
python train_manager.py
echo.
echo Manager Training finished or was interrupted.
echo Press any key to return to the menu.
pause > nul
goto menu

:train_meta
cls
echo ===================================
echo  (L3) Starting AI Training (META-MANAGER / CEO)
echo ===================================
echo.
echo This will train the CEO brain that sets GOALS for the Manager.
echo This requires the Manager (manager_model.zip) to exist first!
echo.
echo META-MANAGER steps are EXTREMELY slow.
This will take a very long time.
echo.
set TF_ENABLE_ONEDNN_OPTS=0
python train_meta.py
echo.
echo Meta-Manager Training finished or was interrupted.
echo Press any key to return to the menu.
pause > nul
goto menu

:watch_specialist
cls
echo ===================================
echo  (L1) Watch a Single AI Specialist Play
echo ===================================
echo.
echo Which model do you want to watch?
echo.
echo --- FINAL MODELS ---
echo 1. exploration_model.zip
echo 2. battle_model.zip
echo 3. healer_model.zip
echo 4. shopping_model.zip
echo 5. inventory_model.zip
echo 6. switch_model.zip
echo.
echo --- SPECIFIC CHECKPOINT ---
echo 7. Watch a specific CHECKPOINT file
echo.
echo 8. Return to Main Menu
echo.
set "model_to_watch="
set "checkpoint_file="
set /p "watch_choice=Enter your choice (1-8): "

if "%watch_choice%"=="1" set "model_to_watch=exploration_model"
if "%watch_choice%"=="2" set "model_to_watch=battle_model"
if "%watch_choice%"=="3" set "model_to_watch=healer_model"
if "%watch_choice%"=="4" set "model_to_watch=shopping_model"
if "%watch_choice%"=="5" set "model_to_watch=inventory_model"
if "%watch_choice%"=="6" set "model_to_watch=switch_model"
if "%watch_choice%"=="7" goto watch_checkpoint
if "%watch_choice%"=="8" goto menu

if not defined model_to_watch (
    echo Invalid choice.
    echo Press any key to return.
    pause > nul
    goto watch_specialist
)

echo Starting AI Watcher (Model: %model_to_watch%.zip)...
echo.
set TF_ENABLE_ONEDNN_OPTS=0
python watch_specialist.py --model_name %model_to_watch%
echo.
echo Watcher finished or was interrupted. Press any key to return to the menu.
pause > nul
goto menu

:watch_checkpoint
cls
echo ===================================
echo  Watch a Specific Checkpoint
echo ===================================
echo.
echo Enter the BASE model name (e.g., exploration_model, battle_model).
set /p "model_to_watch=Base Model Name: "
echo.
echo Now, enter the FULL checkpoint filename you want to watch.
echo (e.g., exploration_model_checkpoint_49994_steps.zip)
set /p "checkpoint_file=Checkpoint Filename: "
echo.

if not defined model_to_watch (
    echo No base model name entered.
    pause > nul
    goto watch_checkpoint
)
if not defined checkpoint_file (
    echo No checkpoint file entered.
    pause > nul
    goto watch_checkpoint
)

echo Starting AI Watcher for Checkpoint...
echo   Base Model: %model_to_watch%
echo   Checkpoint: %checkpoint_file%
echo.
set TF_ENABLE_ONEDNN_OPTS=0
python watch_specialist.py --model_name %model_to_watch% --checkpoint %checkpoint_file%
echo.
echo Watcher finished or was interrupted. Press any key to return to the menu.
pause > nul
goto menu

:watch_rules
echo Starting 'Rules-Based Manager' Watcher (watch_rules_based_manager.py)...
echo This will load ALL specialist models and use hard-coded rules.
echo.
set TF_ENABLE_ONEDNN_OPTS=0
python watch_rules_based_manager.py
echo.
echo Rules-Based Watcher finished or was interrupted. Press any key to return.
pause > nul
goto menu

:watch_manager
echo Starting (L2) 'Manager AI' Watcher (watch_manager.py)...
echo This will load the single 'manager_model.zip'
echo which has learned to control all the specialists.
echo.
set TF_ENABLE_ONEDNN_OPTS=0
python watch_manager.py
echo.
echo Manager Watcher finished or was interrupted. Press any key to return.
pause > nul
goto menu

:watch_meta
echo Starting (L3) 'Meta-Manager AI' Watcher (watch_meta.py)...
echo This will load the full AI stack (CEO -> Manager -> Specialists).
echo.
set TF_ENABLE_ONEDNN_OPTS=0
python watch_meta.py
echo.
echo Meta-Manager Watcher finished or was interrupted. Press any key to return.
pause > nul
goto menu

:tensorboard
echo Starting TensorBoard on http://localhost:6006
echo (This will open a new command window)
echo.
set TF_ENABLE_ONEDNN_OPTS=0
start "TensorBoard" cmd /c "tensorboard --logdir logs --host 0.0.0.0 --port 6006"
echo Press any key to return to the menu.
pause > nul
goto menu

:starter
echo Starting Initial Save State Creator (create_save_state.py)...
echo Follow the instructions in the new window.
echo.
set TF_ENABLE_ONEDNN_OPTS=0
python create_save_state.py
echo.
echo Creator script finished. Press any key to return to the menu.
pause > nul
goto menu

:export_metrics
echo Starting Metrics Exporter (export_metrics.py)...
echo.
set /p "lesson_name=Enter lesson to export (e.g., exploration, manager, meta): "
echo.
echo Running export for 'logs/%lesson_name%'...
echo Output saved to debug_metrics_summary_%lesson_name%.md
echo.
python export_metrics.py --lesson %lesson_name%
echo.
echo Metrics export finished.
echo Press any key to return to the menu.
pause > nul
goto menu

:debug_hash
echo Starting Text Hash Debugger (debug_text_hash.py)...
echo.
echo REQUIRES: pip install keyboard
echo.
echo This script will load 'states/battle_01.state' (or whatever is set in the file).
echo Play the game, and press 'H' in the console to capture the hash of any text on screen.
echo.
set TF_ENABLE_ONEDNN_OPTS=0
python debug_text_hash.py
echo.
echo Debugger finished. Press any key to return to the menu.
pause > nul
goto menu

:debug_scanner
echo Starting GUI Memory Scanner (debug_memory_scanner.py)...
echo.
echo This will open two windows: The Game and the Scanner.
echo Use the buttons on the scanner to find addresses.
echo.
set TF_ENABLE_ONEDNN_OPTS=0
python debug_memory_scanner.py
echo.
echo Scanner finished. Press any key to return to the menu.
pause > nul
goto menu

:debug_memory
echo Starting Memory Watcher (debug_memory_watch.py)...
echo.
echo REQUIRES: pip install keyboard
echo.
echo This script will load 'states/new_game.state'.
echo Play the game and watch the console to see memory values change.
echo Use this to find the real X/Y coordinates!
echo.
set TF_ENABLE_ONEDNN_OPTS=0
python debug_memory_watch.py
echo.
echo Watcher finished.
Press any key to return to the menu.
pause > nul
goto menu

:exit
exit

rem --- DEBUGGING SECTION ---

:debug_specialist
cls
echo ===================================
echo  (L1) Select a DEBUG Specialist Lesson
echo ===================================
echo.
echo 1. Lesson 1: EXPLORATION (DEBUG)
echo 2. Lesson 2: BATTLE (DEBUG)
echo 3. Lesson 3: HEALER (DEBUG)
echo 4. Lesson 4: SHOPPING (DEBUG)
echo 5. Lesson 5: INVENTORY (DEBUG)
echo 6. Lesson 6: SWITCH (DEBUG)
echo.
echo 7. Return to Main Menu
echo.
set /p "debug_choice=Enter your choice (1-7): "

if "%debug_choice%"=="1" goto debug_explore
if "%debug_choice%"=="2" goto debug_battle
if "%debug_choice%"=="3" goto debug_healer
if "%debug_choice%"=="4" goto debug_shopping
if "%debug_choice%"=="5" goto debug_inventory
if "%debug_choice%"=="6" goto debug_switch
if "%debug_choice%"=="7" goto menu

echo Invalid choice.
echo Press any key to return.
pause > nul
goto debug_specialist

:debug_explore
echo Starting AI DEBUG Training (Lesson: EXPLORATION)...
echo.
set TF_ENABLE_ONEDNN_OPTS=0
python train_specialist.py --lesson exploration --model_name exploration_model --debug
echo.
echo DEBUG Training finished. Press any key to return to the menu.
pause > nul
goto menu

:debug_battle
echo Starting AI DEBUG Training (Lesson: BATTLE)...
IF NOT EXIST %STATE_DIR%\battle.state ( IF NOT EXIST %STATE_DIR%\battle_*.state (
    echo.
    echo ERROR: No battle save states found.
    echo.
    pause
    goto menu
))
echo Battle save state(s) found.
echo Starting DEBUG training...
echo.
set TF_ENABLE_ONEDNN_OPTS=0
python train_specialist.py --lesson battle --model_name battle_model --debug
echo.
echo DEBUG Training finished.
echo Press any key to return to the menu.
pause > nul
goto menu

:debug_healer
echo Starting AI DEBUG Training (Lesson: HEALER)...
IF NOT EXIST %STATE_DIR%\healer.state ( IF NOT EXIST %STATE_DIR%\healer_*.state (
    echo.
    echo ERROR: No healer save states found.
    echo.
    pause
    goto menu
))
echo 'healer.state' file found.
echo Starting DEBUG training...
echo.
set TF_ENABLE_ONEDNN_OPTS=0
python train_specialist.py --lesson healer --model_name healer_model --debug
echo.
echo DEBUG Training finished.
echo Press any key to return to the menu.
pause > nul
goto menu

:debug_shopping
echo Starting AI DEBUG Training (Lesson: SHOPPING)...
IF NOT EXIST %STATE_DIR%\shopping.state ( IF NOT EXIST %STATE_DIR%\shopping_*.state (
    echo.
    echo ERROR: No shopping save states found.
    echo.
    pause
    goto menu
))
echo 'shopping.state' file found.
echo Starting DEBUG training...
echo.
set TF_ENABLE_ONEDNN_OPTS=0
python train_specialist.py --lesson shopping --model_name shopping_model --debug
echo.
echo DEBUG Training finished.
echo Press any key to return to the menu.
pause > nul
goto menu

:debug_inventory
echo Starting AI DEBUG Training (Lesson: INVENTORY)...
IF NOT EXIST %STATE_DIR%\inventory.state ( IF NOT EXIST %STATE_DIR%\inventory_*.state (
    echo.
    echo ERROR: No inventory save states found.
    echo.
    pause
    goto menu
))
echo 'inventory.state' file found.
echo Starting DEBUG training...
echo.
set TF_ENABLE_ONEDNN_OPTS=0
python train_specialist.py --lesson inventory --model_name inventory_model --debug
echo.
echo DEBUG Training finished.
echo Press any key to return to the menu.
pause > nul
goto menu

:debug_switch
echo Starting AI DEBUG Training (Lesson: SWITCH)...
IF NOT EXIST %STATE_DIR%\switch.state ( IF NOT EXIST %STATE_DIR%\switch_*.state (
    echo.
    echo ERROR: No switch save states found.
    echo.
    pause
    goto menu
))
echo 'switch.state' file found.
echo Starting DEBUG training...
echo.
set TF_ENABLE_ONEDNN_OPTS=0
python train_specialist.py --lesson switch --model_name switch_model --debug
echo.
echo DEBUG Training finished.
echo Press any key to return to the menu.
pause > nul
goto menu

:debug_manager
echo Starting (L2) AI DEBUG Training (Lesson: MANAGER AI)...
echo.
set TF_ENABLE_ONEDNN_OPTS=0
python train_manager.py --debug
echo.
echo DEBUG Manager Training finished. Press any key to return to the menu.
pause > nul
goto menu

:debug_meta
echo Starting (L3) AI DEBUG Training (Lesson: META-MANAGER AI)...
echo.
set TF_ENABLE_ONEDNN_OPTS=0
python train_meta.py --debug
echo.
echo DEBUG Meta-Manager Training finished. Press any key to return to the menu.
pause > nul
goto menu