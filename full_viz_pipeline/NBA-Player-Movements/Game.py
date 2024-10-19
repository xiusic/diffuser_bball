import pandas as pd
from Event import Event
from Team import Team
from Constant import Constant


class Game:
    """A class for keeping info about the games"""
    def __init__(self, json_df, path_to_json, event_index):
        # self.events = None
        self.home_team = None
        self.guest_team = None
        self.event = None
        self.json_df = json_df
        self.path_to_json = path_to_json
        self.event_index = event_index

    def read_json(self):
        data_frame = self.json_df
        last_default_index = 1
        og_index = self.event_index
        self.event_index = min(self.event_index, last_default_index)
        index = self.event_index

        #print(Constant.MESSAGE + str(last_default_index))
        event = data_frame['events'][index]
        self.home_id = event['home']['teamid']
        self.guest_id = event['visitor']['teamid']
        self.event = Event(self, event, self.path_to_json, og_index)
        self.home_team = Team(self,event['home']['teamid'])
        self.guest_team = Team(self,event['visitor']['teamid'])

    def start(self):
        self.event.show()
