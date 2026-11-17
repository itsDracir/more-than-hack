"""Dades a l'entrada, entrega a la sortida -- l'única ordre que regenera el fitxer que pugem.

    python solution/run.py --name <nom_equip>

Llegeix els dos fitxers que escriu solution/01_preprocessing.ipynb a data/processed/: la taula de
features per frame (features_tactical.csv, velocitats dels equips incloses) i els intervals de
SET PIECES del Smart Tagging (smart_tagging_events.json). No obre ni l'ATD ni l'XML: tota la lectura
de dades en brut es fa una sola vegada al notebook. Etiqueta cada frame, escriu
evaluation/results/<nom_equip>.csv i després hi passa exactament la mateixa comprovació que fa el
scorer oficial.

Determinista: les mateixes dades d'entrada donen un fitxer idèntic byte a byte.
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for path in (ROOT, os.path.join(ROOT, "evaluation")):
    if path not in sys.path:
        sys.path.insert(0, path)

import games  # noqa: E402
import possession_metrics as pm  # noqa: E402
from solution.labeling import label_frames  # noqa: E402
from solution.params import Params  # noqa: E402

SPEED_COLUMNS = ["home_mean_speed_mps", "away_mean_speed_mps"]
DEFAULT_FEATURES = os.path.join(games.DATA_DIR, "processed", "features_tactical.csv")
DEFAULT_EVENTS = os.path.join(games.DATA_DIR, "processed", "smart_tagging_events.json")
RESULTS_DIR = os.path.join(ROOT, "evaluation", "results")

REBUILD = """Build it with the preprocessing notebook, end to end (outputs go to data/processed/,
which git ignores, so no match data ends up in a commit):

    python -m jupyter nbconvert --to notebook --execute solution/01_preprocessing.ipynb --output-dir data/processed

Then run this again, or point {} at another file."""

MISSING_FEATURES = "No feature table at {}\n\n" + REBUILD.format("--features")
MISSING_SPEEDS = ("The feature table {} has no {} column(s): it predates the notebook version that "
                  "computes team speeds.\n\n" + REBUILD.format("--features"))
MISSING_EVENTS = ("No set-piece intervals at {}\n\n" + REBUILD.format("--events")
                  + "\nOr pass --no-smart-tagging to run without them (worse restarts).")


def display_path(path):
    """Ruta relativa al repositori quan hi és a dins, absoluta si no. A Windows relpath falla amb
    una ruta en una altra unitat, i això no ha de fer petar una execució que ja ha acabat."""
    absolute = os.path.abspath(path)
    try:
        relative = os.path.relpath(absolute, ROOT)
    except ValueError:
        return absolute
    return absolute if relative.startswith("..") else relative


def load_features(path, n_frames):
    """La taula de features, una fila per frame de vídeo 0..n_frames-1 en ordre, o un error
    clar."""
    table = pd.read_parquet(path) if path.endswith(".parquet") else pd.read_csv(path, low_memory=False)
    if "frame" not in table.columns:
        raise ValueError("{}: no 'frame' column".format(path))
    table = table.sort_values("frame").reset_index(drop=True)
    if len(table) != n_frames or not np.array_equal(table["frame"].to_numpy(),
                                                    np.arange(n_frames)):
        raise ValueError("{}: expected one row per video frame 0..{}, got {} rows"
                         .format(path, n_frames - 1, len(table)))
    if "ball_visible" in table.columns and table["ball_visible"].dtype != bool:
        table["ball_visible"] = table["ball_visible"].astype(str).str.lower() == "true"
    return table


def missing_speed_columns(features):
    """Les columnes de velocitat per equip que falten a la taula (llista buida si hi són)."""
    return [c for c in SPEED_COLUMNS if c not in features.columns]


def load_events(path=DEFAULT_EVENTS):
    """Intervals de SET PIECES que el notebook ha desat: [{code, start_frame, end_frame (exclòs),
    tags}], el mateix format que starter/load_smart_tagging.load()."""
    with open(path, encoding="utf-8") as handle:
        events = json.load(handle)
    if not isinstance(events, list) or not all(
            isinstance(e, dict) and {"code", "start_frame", "end_frame", "tags"} <= set(e)
            for e in events):
        raise ValueError("{}: expected a list of {{code, start_frame, end_frame, tags}}"
                         .format(display_path(path)))
    return events


def write_submission(path, labels):
    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)
    with open(path, "w", newline="\n") as handle:
        handle.write("frame,label\n")
        handle.writelines("{},{}\n".format(frame, label) for frame, label in enumerate(labels))
    return path


def summarise(labels, blocks):
    """El que cal revisar abans de pujar-lo -- calculat sobre el nostre fitxer, sense ground
    truth."""
    runs = [(first, last + 1) for _, first, last in blocks]
    graded = np.concatenate([labels[a:b] for a, b in runs])
    print("\ngraded frames: {}".format(len(graded)))
    for label in "HADX":
        count = int(np.sum(graded == label))
        print("  {}  {:>7}  {:5.1f}%".format(label, count, 100.0 * count / len(graded)))
    codes = pm.fill_abstentions(pm.label_codes(labels), runs)
    stats = pm.possession_stats(codes, runs)
    print("possessions: {}  mean {:.1f} s  home share {:.1%}".format(
        stats["n_spells"], stats["mean_duration_s"], stats["home_share"]))
    print("(sanity check, not a target: D is ball-out-of-play time, roughly 35-45% of a "
          "professional match; a D share far outside that means the dead-ball signal is off.)")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--name", required=True, help="team name: the file is <name>.csv")
    parser.add_argument("--features", default=DEFAULT_FEATURES, help="per-frame feature table")
    parser.add_argument("--out", default=RESULTS_DIR)
    parser.add_argument("--events", default=DEFAULT_EVENTS,
                        help="set-piece intervals written by the preprocessing notebook")
    parser.add_argument("--no-smart-tagging", action="store_true")
    args = parser.parse_args(argv)

    if not os.path.exists(args.features):
        print(MISSING_FEATURES.format(display_path(args.features)), file=sys.stderr)
        return 2

    n_frames = games.source_frames("tactical")
    blocks = games.half_video_blocks(games.GAME)
    features = load_features(args.features, n_frames)
    missing = missing_speed_columns(features)
    if missing:
        print(MISSING_SPEEDS.format(display_path(args.features), ", ".join(missing)),
              file=sys.stderr)
        return 2
    events = None
    if not args.no_smart_tagging:
        if not os.path.exists(args.events):
            print(MISSING_EVENTS.format(display_path(args.events)), file=sys.stderr)
            return 2
        events = load_events(args.events)

    labels = label_frames(features, events, blocks, params=Params())
    path = write_submission(os.path.join(args.out, args.name + ".csv"), labels)

    problem = pm.validate_submission(pm.read_label_csv(path), n_frames)
    print("\n{}  {}".format("INVALID: " + problem if problem else "valid",
                            display_path(path)))
    summarise(labels, blocks)
    print("\nscore on the public window:  python evaluation/run_evaluation.py")
    return 1 if problem else 0


if __name__ == "__main__":
    sys.exit(main())
