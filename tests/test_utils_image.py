__author__ = 'Sergey Matyunin'

import unittest
import utils.image


class TestSequence(unittest.TestCase):
    def setUp(self):
        self.start = 1
        self.len_seq = 20
        self.end = self.start + self.len_seq

        list_paths = ["data/input/sintel/final/cave_2/frame_{:04d}.png".format(i) for i in range(self.start, self.end)]
        self.images = utils.image.load_image_sequence(list_paths)


class TestCrop(TestSequence):
    def test_crop(self):
        cropped = utils.image.crop_image_sequence(self.images, slice(55, 71), slice(20, 30))
        for img in cropped:
            self.assertEqual(cropped[0].shape, img.shape)
        self.assertEqual(cropped[0].shape, (16, 10, 3))

    def test_empty_crop(self):
        cropped = utils.image.crop_image_sequence(self.images, slice(None), slice(None))
        for img in cropped:
            self.assertEqual(cropped[0].shape, img.shape)
        self.assertEqual(cropped[0].shape, self.images[0].shape)


class TestLoadSequence(TestSequence):
    def test_len(self):
        self.assertEqual(len(self.images), self.len_seq)

    def test_same_size(self):
        for img in self.images:
            self.assertEqual(self.images[0].shape, img.shape)

    def test_size(self):
        self.assertEqual(self.images[0].shape, (436, 1024, 3))

    def test_type(self):
        self.assertTrue(isinstance(self.images, list))


class TestSequenceFail(unittest.TestCase):
    def test_too_long(self):
        with self.assertRaises(Exception):
            list_paths = ["data/input/sintel/final/cave_2/frame_{:04d}.png".format(i) for i in range(1, 1000)]
            utils.image.load_image_sequence(list_paths)

    def test_non_exists(self):
        with self.assertRaises(Exception):
            list_paths = ["data/input/sintel/final/cave_2/frame_{:04d}.png".format(i) for i in range(-10, 1)]
            utils.image.load_image_sequence(list_paths)


if __name__ == '__main__':
    unittest.main()
