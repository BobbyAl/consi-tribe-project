import os
import numpy as np
import pandas as pd
from multiprocessing import Process

CACHE_FOLDER = "./cache"
OUTPUT_DIR = "brain_responses_1hz"
BATCH_SIZE = 25  # frames per subprocess — tune down if still crashing
DPI = 100


def compute_indices(cache_folder):
    segments_df = pd.read_parquet(f"{cache_folder}/segments.parquet")
    start_times = segments_df["start"].values
    return np.array([np.argmin(np.abs(start_times - t)) for t in np.arange(325)])


def render_batch(batch_start, batch_end, indices_1hz, cache_folder, output_dir):
    """
    Runs in a subprocess. All memory — including nilearn/matplotlib module-level
    caches — is freed by the OS when this process exits.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from tribev2.plotting import PlotBrain
    import gc

    preds = np.load(f"{cache_folder}/predictions.npy")
    segments_df = pd.read_parquet(f"{cache_folder}/segments.parquet")
    segments = [row.to_dict() for _, row in segments_df.iterrows()]

    plotter = PlotBrain(mesh="fsaverage5")

    for i in range(batch_start, batch_end):
        file_path = os.path.join(output_dir, f"response_{i:03d}.png")
        if os.path.exists(file_path):
            continue

        idx = indices_1hz[i]
        fig = plotter.plot_timesteps(
            preds[idx:idx+1],
            segments=[segments[idx]],
            cmap="fire",
            norm_percentile=99,
            vmin=.6,
            alpha_cmap=(0, .2),
            show_stimuli=False
        )
        fig.savefig(file_path, bbox_inches="tight", pad_inches=0, dpi=DPI)
        plt.close("all")
        gc.collect()


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    indices_1hz = compute_indices(CACHE_FOLDER)
    print(f"Downsampled to 325 frames (1Hz). Starting render...")

    for batch_start in range(0, 325, BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, 325)

        # Check if batch is already fully done
        if all(
            os.path.exists(os.path.join(OUTPUT_DIR, f"response_{i:03d}.png"))
            for i in range(batch_start, batch_end)
        ):
            print(f"  [{batch_start}-{batch_end}) already done, skipping.")
            continue

        print(f"  Rendering frames {batch_start}-{batch_end}...")
        p = Process(
            target=render_batch,
            args=(batch_start, batch_end, indices_1hz, CACHE_FOLDER, OUTPUT_DIR),
        )
        p.start()
        p.join()

        if p.exitcode != 0:
            print(f"  WARNING: batch {batch_start}-{batch_end} exited with code {p.exitcode}")
        else:
            done = len(os.listdir(OUTPUT_DIR))
            print(f"  Done. Total saved: {done}/325")

    print(f"Export complete. {len(os.listdir(OUTPUT_DIR))} images saved.")


if __name__ == "__main__":
    main()
