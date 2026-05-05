import argparse

from isaaclab.app import AppLauncher

# Add command line arguments
parser = argparse.ArgumentParser(description="Load USD model and view joint information")
parser.add_argument(
    "--usd_path",
    type=str,
    default=None,
    help="Path to USD model file (relative to assets directory or absolute path)",
)
app = AppLauncher(headless=True).app

args_cli = parser.parse_args()

from pxr import Sdf

layer = Sdf.Layer.FindOrOpen(args_cli.usd_path)
print("layer", bool(layer))

if layer:
    text = layer.ExportToString()
    print(text)
    for line in text.splitlines():
        if "@" in line or "references" in line or "payload" in line:
            print(line)

app.close()