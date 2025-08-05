
# Build executable without make

poetry run python -m PyInstaller --onefile --add-data "KPIS\template.html;KPIS\\" configurable_main.py

mkdir -p dist/
cp config.yaml dist/config.yaml

