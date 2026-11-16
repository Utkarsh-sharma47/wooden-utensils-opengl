"""Optional Linux/EGL integration test. Run from any directory."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import moderngl
from PIL import Image
import scene

out=scene.ROOT/'test-output'
out.mkdir(exist_ok=True)
ctx=moderngl.create_standalone_context(require=330, backend='egl')
r=scene.Renderer(ctx,384,384)
images=[]
for mode in [0,1]:
    r.shadow_mode=mode
    r.render()
    ctx.finish()
    path=out/('pcf.png' if mode else 'hard.png')
    r.save(path)
    a=np.array(Image.open(path)).astype(float)
    assert a.std()>10, 'Render is blank or almost uniform'
    images.append(a)
assert np.abs(images[0]-images[1]).mean()>.01, 'Shadow modes did not differ'
print('PASS: OpenGL', ctx.info['GL_VERSION'])
print('Renderer:', ctx.info['GL_RENDERER'])
print('Actual render outputs:', out)
