import ssbm
from state import *
import state_manager
import memory_watcher
from menu_manager import *
import os
from pad import *
import time
import util
from numpy import random
import movie
from default import *


MENU_SELECTION_STATE = 2
POSTGAME_STATE = 3
RESETTING_MATCH_STATE = 4

class CPU(Default):
    _options = [
      Option('tag', type=int),
      Option('user', type=str, help="dolphin user directory"),
      Option('zmq', type=bool, default=True, help="use zmq for memory watcher"),
      Option('stage', type=str, default="final_destination", choices=movie.stages.keys(), help="which stage to play on"),
      Option('enemy', type=str, help="load enemy agent from file"),
      Option('enemy_reload', type=int, default=0, help="enemy reload interval"),
      Option('cpu', type=int, help="enemy cpu level"),
      Option('p1', type=str, choices=characters.keys(), default="marth", help="character for player 1"),
      Option('p2', type=str, choices=characters.keys(), default="fox", help="character for player 2"),
    ]

    def __init__(self, **kwargs):
        Default.__init__(self, **kwargs)

        self.toggle = False

        self.user = os.path.expanduser(self.user)

        self.state = ssbm.GameMemory()
        # track players 1 and 2 (pids 0 and 1)
        self.sm = state_manager.StateManager([0, 1])
        self.write_locations()

        if self.tag is not None:
            random.seed(self.tag)

        self.pids = [1]
        self.cpus = {1: None}
        self.characters = {1: self.p2}

        if not self.cpu:
            raise Exception('Expected to play against CPU!.')

        self.pids.append(0)
        self.cpus[0] = None # Controlled by this program.
        self.characters[0] = self.p1

        print('Creating MemoryWatcher.')
        mwType = memory_watcher.MemoryWatcher
        if self.zmq:
          mwType = memory_watcher.MemoryWatcherZMQ
        self.mw = mwType(self.user + '/MemoryWatcher/MemoryWatcher')
        
        pipe_dir = self.user + '/Pipes/'
        print('Creating Pads at %s. Open dolphin now.' % pipe_dir)
        util.makedirs(self.user + '/Pipes/')
        
        paths = [pipe_dir + 'phillip%d' % i for i in self.pids]
        self.get_pads = util.async_map(Pad, paths)

        self.init_stats()

    def run(self, frames=None, dolphin_process=None):
        try:
            self.pads = self.get_pads()
        except KeyboardInterrupt:
            print("Pipes not initialized!")
            return
        
        pick_chars = []
        
        tapA = [
            (0, movie.pushButton(Button.A)),
            (0, movie.releaseButton(Button.A)),
        ]

        enter_stage_select = [
            (28, movie.pushButton(Button.START)),
            (1, movie.releaseButton(Button.START)),
            (10, movie.neutral)
        ]

        for pid, pad in zip(self.pids, self.pads):
            actions = []
            
            cpu = self.cpus[pid]
            
            if cpu:
                actions.append(MoveTo([0, 20], pid, pad, True))
                actions.append(movie.Movie(tapA, pad))
                actions.append(movie.Movie(tapA, pad))
                actions.append(MoveTo([0, -14], pid, pad, True))
                actions.append(movie.Movie(tapA, pad))
                actions.append(MoveTo([cpu * 1.1, 0], pid, pad, True))
                actions.append(movie.Movie(tapA, pad))
                #actions.append(Wait(10000))
            
            actions.append(MoveTo(characters[self.characters[pid]], pid, pad))
            actions.append(movie.Movie(tapA, pad))
            
            pick_chars.append(Sequential(*actions))
        
        pick_chars = Parallel(*pick_chars)
        
        enter_settings = Sequential(
            MoveTo(settings, self.pids[0], self.pads[0]),
            movie.Movie(tapA, self.pads[0])
        )
        
        # sets the game mode and picks the stage
        # start_game = movie.Movie(movie.endless_netplay + movie.stages[self.stage], self.pads[0])
        start_game = movie.Movie(enter_stage_select + movie.stages[self.stage], self.pads[0])

        # self.navigate_menus = Sequential(pick_chars, enter_settings, start_game)
        self.navigate_menus = Sequential(pick_chars, start_game)

    def init_stats(self):
        self.total_frames = 0
        self.skip_frames = 0
        self.thinking_time = 0

    def print_stats(self):
        total_time = time.time() - self.start_time
        frac_skipped = self.skip_frames / self.total_frames
        frac_thinking = self.thinking_time * 1000 / self.total_frames
        print('Total Time:', total_time)
        print('Total Frames:', self.total_frames)
        print('Average FPS:', self.total_frames / total_time)
        print('Fraction Skipped: {:.6f}'.format(frac_skipped))
        print('Average Thinking Time (ms): {:.6f}'.format(frac_thinking))

    def write_locations(self):
        path = self.user + '/MemoryWatcher/'
        util.makedirs(path)
        print('Writing locations to:', path)
        with open(path + 'Locations.txt', 'w') as f:
            f.write('\n'.join(self.sm.locations()))

    def advance_frame(self, reset_match=False):
        last_frame = self.state.frame
        
        self.update_state()
        match_state = None
        menu_state = None
        if self.state.frame > last_frame:
            skipped_frames = self.state.frame - last_frame - 1
            if skipped_frames > 0:
                self.skip_frames += skipped_frames
            self.total_frames += self.state.frame - last_frame
            last_frame = self.state.frame

            start = time.time()
            match_state, menu_state = self.make_action(reset_match)
            if match_state is not None and skipped_frames > 0:
                print("Skipped match frames ", skipped_frames)
            self.thinking_time += time.time() - start

            # if self.state.frame % (15 * 60) == 0:
            #     self.print_stats()
        
        self.mw.advance()
        return match_state, menu_state

    def update_state(self):
        messages = self.mw.get_messages()
        for message in messages:
          self.sm.handle(self.state, *message)
    
    def spam(self, buttons, pad_indexes=None):
        pad_indexes = pad_indexes if pad_indexes is not None else [0]
        for pad_index in pad_indexes:
            self.pads[pad_index].tilt_stick(Stick.MAIN, 0.5, 0.5)
            if self.toggle:
                for button in Button:
                    if button in buttons:
                        self.pads[pad_index].press_button(button)
                    else:
                        self.pads[pad_index].release_button(button)
            else:
                for button in Button:  # Release all buttons.
                    self.pads[pad_index].release_button(button)


        self.toggle = not self.toggle

    # Returns match_state (ssbm.SimpleStateAction), menu_state (number)
    # One and only one of the returned tuple elements is None.
    def make_action(self, reset_match):
        # menu = Menu(self.state.menu)
        # print(menu)
        if self.state.menu == Menu.Game.value:
            if reset_match:
                self.spam([Button.START, Button.A, Button.L, Button.R])
                return None, RESETTING_MATCH_STATE

            return self.state, None

        elif self.state.menu in [menu.value for menu in [Menu.Characters, Menu.Stages]]:
            self.navigate_menus.move(self.state)

            if self.navigate_menus.done():
                for pid, pad in zip(self.pids, self.pads):
                    if self.characters[pid] == 'sheik':
                        pad.press_button(Button.A)

            return None, MENU_SELECTION_STATE

        elif self.state.menu == Menu.PostGame.value:
            self.spam([Button.START], pad_indexes=[0,1])
            stage_select = [
                            (28, movie.pushButton(Button.START)),
                            (1, movie.releaseButton(Button.START)),
                            (10, movie.neutral),
                            (0, movie.tiltStick(Stick.MAIN, 1, 0.8)),
                            (5, movie.tiltStick(Stick.MAIN, 0.5, 0.5)),
                            (20, movie.pushButton(Button.START)),
                            (1, movie.releaseButton(Button.START)),
                            (0, movie.tiltStick(Stick.MAIN, 1, 0.8)),
                            (5, movie.tiltStick(Stick.MAIN, 0.5, 0.5)),
                            (20, movie.pushButton(Button.START)),
                            (1, movie.releaseButton(Button.START))]
            self.navigate_menus = Sequential(movie.Movie(stage_select,self.pads[0]))

            return None, POSTGAME_STATE

        else:
            print("Weird menu state", self.state.menu)
            return None, POSTGAME_STATE


def runCPU(**kwargs):
  CPU(**kwargs).run()

