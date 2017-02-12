import sarsa
import unittest

def fun(x):
  return x + 1

class FullModelTest(unittest.TestCase):
  def setUp(self):
    self.full_model = sarsa.FullModel()


  def testCoordinateToState(self):
    self.assertEquals(self.full_model.coordinate_to_state(0, 0), 71)
    self.assertEquals(self.full_model.coordinate_to_state(1, 0), 71)
    self.assertEquals(self.full_model.coordinate_to_state(0, 1), 71)

    self.assertEquals(self.full_model.coordinate_to_state(10, 0), 81)
    self.assertEquals(self.full_model.coordinate_to_state(0, 10), 72)
    self.assertEquals(self.full_model.coordinate_to_state(10, 50), 86)
    self.assertEquals(self.full_model.coordinate_to_state(50, 10), 122)
    self.assertEquals(self.full_model.coordinate_to_state(-50, 10), 22)

    # Clipping.
    self.assertEquals(self.full_model.coordinate_to_state(-100, 0), 1)
    self.assertEquals(self.full_model.coordinate_to_state(0, -100), 70)
    self.assertEquals(self.full_model.coordinate_to_state(-100, -100), 0)

    self.assertEquals(self.full_model.coordinate_to_state(100, 0), 131)
    self.assertEquals(self.full_model.coordinate_to_state(0, 100), 79)
    self.assertEquals(self.full_model.coordinate_to_state(100, 100), 139)
