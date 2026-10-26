#Adapt MMCV 1.x GPU scatter streams to PyTorch 2.x's device API. Stemms from https://github.com/nabe2030/bevformer-blackwell
from pathlib import Path
import sysconfig

path = Path(sysconfig.get_paths()['purelib']) / 'mmcv/parallel/_functions.py'
old = 'streams = [_get_stream(device) for device in target_gpus]'
new = 'streams = [_get_stream(torch.device("cuda", device) if isinstance(device, int) else device) for device in target_gpus]'
source = path.read_text()
if old in source:
    path.write_text(source.replace(old, new))