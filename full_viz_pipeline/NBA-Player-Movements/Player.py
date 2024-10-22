from Team import Team


class Player:
    """A class for keeping info about the players"""
    def __init__(self,game, player):
        self.team = Team(game,player[0])
        self.id = player[1]
        self.x = player[2]
        self.y = player[3]
        self.color = self.team.color
