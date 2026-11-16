import unittest
import numpy as np
from PIL import Image
import scene

class GeometryTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.meshes = scene.make_scene()

    def test_meshes(self):
        self.assertGreater(len(self.meshes), 100)
        for mesh in self.meshes:
            with self.subTest(mesh=mesh.name):
                self.assertEqual(mesh.v.shape[1], 3)
                self.assertEqual(mesh.f.shape[1], 3)
                self.assertTrue(np.isfinite(mesh.v).all())
                self.assertTrue(np.isfinite(mesh.n).all())
                self.assertTrue(np.isfinite(mesh.model).all())
                self.assertGreaterEqual(mesh.f.min(), 0)
                self.assertLess(mesh.f.max(), len(mesh.v))

    def test_single_spoon(self):
        meshes=scene.make_scene('single')
        self.assertEqual({m.name for m in meshes}, {'beige tabletop','spoon handle','spoon head'})

    def test_wood_assets(self):
        for name in ['wood-color.jpg', 'wood-rough.jpg', 'wood-normal.jpg']:
            with Image.open(scene.ROOT / name) as im:
                self.assertEqual(im.size, (1024, 1024))
                im.verify()

if __name__ == '__main__':
    unittest.main()
