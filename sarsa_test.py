import sarsa
import unittest


class FullModelTest(unittest.TestCase):
  def setUp(self):
    self.full_model = sarsa.FullModel()


  def testToggleExploration(self):
    # TODO(bparr): Don't look at protected internal state and instead test
    #              functionality.
    self.assertTrue(self.full_model.explore_on)

    self.full_model.toggle()
    self.assertFalse(self.full_model.explore_on)

    self.full_model.toggle()
    self.assertTrue(self.full_model.explore_on)


  def testAct(self):
    # TODO(bparr): Use self._epsilon_greedy_fn in original code so can inject
    #              a mock function here and thus easily test exploration and
    #              non exploration steps.
    pass


  def testRewardWhenNearCenter(self):
    self.assertEquals(self.full_model.reward(0, 0), 1)
    self.assertEquals(self.full_model.reward(19, 0), 1)
    self.assertEquals(self.full_model.reward(0, 19), 1)
    self.assertEquals(self.full_model.reward(10, 10), 1)


  def testRewardWhenOffStage(self):
    self.assertEquals(self.full_model.reward(-61, 0), -1)
    self.assertEquals(self.full_model.reward(61, 0), -1)
    self.assertEquals(self.full_model.reward(61, 42), -1)


  def testRewardWhenOnStageButNotNearCenter(self):
    self.assertEquals(self.full_model.reward(21, 0), 0)
    self.assertEquals(self.full_model.reward(0, 21), 0)
    self.assertEquals(self.full_model.reward(16, 16), 0)


  def testUpdate(self):
    # TODO(bparr): Implement. I think we need to discuss how best to do this.
    pass


  def testCoordinateToState(self):
    self.assertEquals(self.full_model.coordinate_to_state(0, 0), 71)
    self.assertEquals(self.full_model.coordinate_to_state(1, 0), 71)
    self.assertEquals(self.full_model.coordinate_to_state(0, 1), 71)

    self.assertEquals(self.full_model.coordinate_to_state(10, 0), 81)
    self.assertEquals(self.full_model.coordinate_to_state(0, 10), 72)
    self.assertEquals(self.full_model.coordinate_to_state(10, 50), 86)
    self.assertEquals(self.full_model.coordinate_to_state(50, 10), 122)
    self.assertEquals(self.full_model.coordinate_to_state(-50, 10), 22)

  def testCoordinateToStateClipping(self):
    self.assertEquals(self.full_model.coordinate_to_state(-100, 0), 1)
    self.assertEquals(self.full_model.coordinate_to_state(0, -100), 70)
    self.assertEquals(self.full_model.coordinate_to_state(-100, -100), 0)

    self.assertEquals(self.full_model.coordinate_to_state(100, 0), 131)
    self.assertEquals(self.full_model.coordinate_to_state(0, 100), 79)
    self.assertEquals(self.full_model.coordinate_to_state(100, 100), 139)
