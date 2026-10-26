"""Execute a notebook headlessly, saving it after every cell and streaming printed output to a log.

usage: python run_notebook.py NOTEBOOK [--out OUT.ipynb] [--log LOG.txt] [--override "PY CODE"]
--override inserts a code cell right after the first cell that defines DEVICE (for smoke tests).
Runs with the notebook's folder as the working directory, like VS Code / JupyterLab.
"""
import argparse, time
from pathlib import Path
import nbformat
from nbclient import NotebookClient

ap = argparse.ArgumentParser()
ap.add_argument('notebook'); ap.add_argument('--out'); ap.add_argument('--log'); ap.add_argument('--override')
args = ap.parse_args()
src = Path(args.notebook).resolve()
out = Path(args.out).resolve() if args.out else src
log = open(args.log or out.with_suffix('.log'), 'a', buffering=1)
nb = nbformat.read(src, as_version=4)
if args.override:
    at = next(i for i, c in enumerate(nb.cells) if c.cell_type == 'code' and 'DEVICE =' in c.source)
    nb.cells.insert(at + 1, nbformat.v4.new_code_cell('# SMOKE-TEST OVERRIDE\n' + args.override))


class LoggingClient(NotebookClient):
    def output(self, outs, msg, display_id, cell_index):
        c = msg['content']
        if msg['msg_type'] == 'stream':
            log.write(c['text'])
        elif msg['msg_type'] == 'error':
            log.write(f"ERROR {c['ename']}: {c['evalue']}\n")
        return super().output(outs, msg, display_id, cell_index)


def saved(cell, cell_index, **_):
    nbformat.write(nb, out)
    log.write(f'[{time.strftime("%H:%M:%S")}] cell {cell_index} done\n')


client = LoggingClient(nb, timeout=None, kernel_name='python3', allow_errors=False,
                       resources={'metadata': {'path': str(src.parent)}}, on_cell_executed=saved)
log.write(f'[{time.strftime("%H:%M:%S")}] start {src.name}\n')
try:
    client.execute()
    log.write(f'[{time.strftime("%H:%M:%S")}] FINISHED OK\n')
except Exception as exc:
    log.write(f'[{time.strftime("%H:%M:%S")}] FAILED: {type(exc).__name__}: {str(exc)[-2000:]}\n')
    raise
finally:
    nbformat.write(nb, out)
