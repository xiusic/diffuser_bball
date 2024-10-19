import argparse
import sys
sys.path.append('/full_viz_pipeline/NBA-Player-Movements')
from gif_visual_pipeline import visualize


# add arguments path and index to pass to script:
parser = argparse.ArgumentParser(description='Process arguments for npy filepath and traj_index.')
parser.add_argument('--path', type=str,
                    help='a path to json file to read the events from',
                    required = True)
parser.add_argument('--index', type=int,
                    help="""an index of the event to create the animation to
                            (the indexing start with zero, if you index goes beyond out
                            the total number of events (plays), it will show you the last
                            one of the game)""")

args = parser.parse_args()

# create json file from the given numpy and traj

visualize(npy_file=args.path)


 
