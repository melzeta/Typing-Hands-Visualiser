# Typing-Hands-Visualiser: Real-Time 3D Motion & Velocity Analysis

This tool is a real-time visualization utility for analyzing typing kinematics, specifically designed for datasets containing 3D joint coordinates for both the left and right hands. It's built with **Tkinter** for the UI and **Matplotlib** for embedded visualization.

---

## Key Features & Visualization

The interface is split into two main panes:

* **3D Hand Skeletons:** Renders both hands. The **selected hand** (for analysis) is highlighted with color-coded fingers, while the other hand remains a neutral grey for contextual reference.
* **Per-Finger Velocity Plots:** Displays **live speed curves** for the fingertips (Thumb, Index, Middle, Ring, Little) of the selected hand. These plots show a sliding window of the **last 10 seconds** of movement, with speeds computed frame-by-frame from raw joint positions.
* **Dataset Control:** A dropdown allows selection of the primary hand ("Left" or "Right"). A "Swap L/R datasets" toggle is also provided to quickly invert the dataset used for analysis and highlighting without altering the underlying coordinate data. **Switching resets the velocity plots and the 10-second time window.**

---

## Data Input Format

The script requires a **CSV file** containing 3D joint coordinates.

### Column Naming Convention
Columns must adhere to the pattern: `Hands_<L or R>_<JointName>_<x|y|z>`.

| Example Columns | Description |
| :--- | :--- |
| `Hands_L_Win_x`, `Hands_L_Win_y`, `Hands_L_Win_z` | Left Hand Joint |
| `Hands_R_T4_x`, `Hands_R_T4_y`, `Hands_R_T4_z` | Right Hand Joint |

### Time Handling
* The script automatically detects and uses a numeric **time column** if present. It converts likely millisecond-based timestamps to seconds.
* If no time column is found, it falls back to using a **constant frame rate** (`TARGET_FPS`) for velocity calculations.

---

## Core Configuration

Key operational settings are easily accessible at the top of the main script:

| Setting | Default Value | Description |
| :--- | :--- | :--- |
| `CSV_PATH` | `"cleaned_data_coordinates_216367.csv"` | Path to the input data file. |
| `TARGET_FPS` | `30` | Fallback frame rate (frames per second). |
| `WINDOW_SEC` | `10.0` | Duration of the sliding time window for speed plots (seconds). |
| `VELOCITY_UNIT` | `"m/s"` | Display unit for velocity plots. |

---

## Execution Pipeline

The tool operates via a straightforward frame-based animation loop:

1.  **Data Ingestion:** Loads the CSV file, identifies and structures all required joint columns.
2.  **Skeleton Logic:** Establishes the connection hierarchy (palm, finger chains) for 3D rendering.
3.  **UI Initialization:** Sets up the Tkinter window and embeds the Matplotlib canvas.
4.  **Animation Loop:** Iterates through data rows, performing these steps per frame:
    * Reads the next set of joint coordinates.
    * Updates the 3D skeleton visualization for both hands.
    * Calculates **fingertip velocities** for the selected hand.
    * Appends the new velocity data to the **10-second sliding buffer**.
    * Dynamically **rescales the speed plots** based on recent peak velocity values for optimal visibility.

---

## Getting Started

### Requirements
You'll need a standard Python environment with these libraries:
* **Python 3**
* **numpy**
* **pandas**
* **matplotlib**
* **tkinter** (usually included with Python)

### Run Command
Execute the script from your terminal:

```bash
python viz_rt.py
