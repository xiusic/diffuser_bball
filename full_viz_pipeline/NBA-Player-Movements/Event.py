from Constant import Constant
from Moment import Moment
from Team import Team
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import animation
from matplotlib.patches import Circle, Rectangle, Arc
from tqdm import tqdm
import os

class Event:
    """A class for handling and showing events"""

    def __init__(self, game, event, fname, event_index):
        self.event_index = event_index
        self.fname = fname
#        print(self.fname)
        moments = event['moments']
        self.moments = [Moment(game,moment) for moment in moments]
        home_players = event['home']['players']
        guest_players = event['visitor']['players']
        players = home_players + guest_players
        player_ids = [player['playerid'] for player in players]
        player_names = [" ".join([player['firstname'],
                        player['lastname']]) for player in players]
        player_jerseys = [player['jersey'] for player in players]
        values = list(zip(player_names, player_jerseys))
        # Example: 101108: ['Chris Paul', '3']
        self.player_ids_dict = dict(zip(player_ids, values))

    def update_radius(self, i, player_circles, ball_circle, annotations, clock_info):
        moment = self.moments[i]
        for j, circle in enumerate(player_circles):
            circle.center = moment.players[j].x, moment.players[j].y
            annotations[j].set_position(circle.center)
            clock_test = 'Quarter {:d}\n {:02d}:{:02d}\n {:03.1f}'.format(
                         moment.quarter,
                         int(moment.game_clock) % 3600 // 60,
                         int(moment.game_clock) % 60,
                         moment.shot_clock)
            #clock_info.set_text(clock_test)
        ball_circle.center = moment.ball.x, moment.ball.y
        ball_circle.radius = moment.ball.radius / Constant.NORMALIZATION_COEF
        return player_circles, ball_circle

    def show(self):
        """
        """
#        print("HERE")
        # Leave some space for inbound passes
        ax = plt.axes(xlim=(Constant.X_MIN,
                            Constant.X_MAX),
                      ylim=(Constant.Y_MIN,
                            Constant.Y_MAX))
        ax.axis('off')
        fig = plt.gcf()
        ax.grid(False)  # Remove grid
        start_moment = self.moments[0]
        player_dict = self.player_ids_dict
        #player_list = [player.id for player in start_moment.players]
        #arbvalues = ['1','2','3','4','5','1','2','3','4','5']
        #player_dict_jerseys = {p: v for p, v in zip(player_list, arbvalues)}

        clock_info = ax.annotate('', xy=[Constant.X_CENTER, Constant.Y_CENTER],
                                 color='black', horizontalalignment='center',
                                   verticalalignment='center')

        annotations = [ax.annotate(None, xy=[0, 0], color='w',
                                   horizontalalignment='center',
                                   verticalalignment='center', fontweight='bold')
                       for player in start_moment.players]

        # Prepare table
        sorted_players = sorted(start_moment.players, key=lambda player: player.team.id)
        
        home_player = sorted_players[0]
        guest_player = sorted_players[5]
        column_labels = tuple([home_player.team.name, guest_player.team.name])
        column_colours = tuple([home_player.team.color, guest_player.team.color])
        cell_colours = [column_colours for _ in range(5)]
        
        home_players = [' #'.join([player_dict[player.id][0], player_dict[player.id][1]]) for player in sorted_players[:5]]
        guest_players = [' #'.join([player_dict[player.id][0], player_dict[player.id][1]]) for player in sorted_players[5:]]
        players_data = list(zip(home_players, guest_players))


        player_circles = [plt.Circle((0, 0), Constant.PLAYER_CIRCLE_SIZE, color=player.color)
                          for player in start_moment.players]
        ball_circle = plt.Circle((0, 0), Constant.PLAYER_CIRCLE_SIZE,
                                 color=start_moment.ball.color)
        for circle in player_circles:
            ax.add_patch(circle)
        ax.add_patch(ball_circle)

        anim = animation.FuncAnimation(
                         fig, self.update_radius,
                         fargs=(player_circles, ball_circle, annotations, clock_info),
                         frames=len(self.moments), interval=Constant.INTERVAL)


        zone_boundaries = [
        (26, 47, 63, 81),   # Zone 0 (on 3 side)
        (3, 24, 63, 81),    # Zone 1 (on 4 side)
        (31, 49, 81, 93),   # Zone 2
        (1, 19, 81, 93),    # Zone 3
        (19, 31, 75, 94)    # Zone 4
        ]

        # Plot highlighted yellow rectangles around the zone boundaries (COMMENT THIS IF NOT USING 2_3 HUERISTICS)

        # for zone_index, (y_min, y_max, x_min, x_max) in enumerate(zone_boundaries):
        #     rect = patches.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, linewidth=2, edgecolor='black', facecolor='yellow', alpha = 0.3, label=f'Zone {zone_index}')
        #     ax.add_patch(rect)



        court = plt.imread("/NBA-Player-Movements/court.png")
        plt.imshow(court, zorder=0, extent=[Constant.X_MIN, Constant.X_MAX - Constant.DIFF,
                                            Constant.Y_MAX, Constant.Y_MIN])


        save_dir = "/test_gifs/"

        path_parts = self.fname.split('/logs/')
        subfolder, filename = path_parts[1].split('/', 1)

        save_folder = os.path.join(save_dir, subfolder)


        # Create the target directory if it doesn't exist
        os.makedirs(save_folder, exist_ok=True)

        basefilename = os.path.basename(self.fname)
        name, extension = os.path.splitext(basefilename)
        if extension:
            basefilename = name + '.gif'
        
        save_fname = os.path.join(save_folder, basefilename)
        print(save_fname)
        anim.save(save_fname, writer='pillow')
        #print("saved.")
