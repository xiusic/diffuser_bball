code for the working paper "Planning with Diffusion for Professional Basketball Player Behavior Synthesis."


Hueristic problem - to only do this for one team, we need to generate trajectories for all teams, then in post update the defending teams position to be that of hueristics, we do this in several different number of batches


Types of hueristics:
- Original - Defense players trail offensive players only allowing 1 defense player per offensive player. When close enough, defensive players always manage to be in front of the opposing players.
- Loose - Defensive players trail offensive players only allowing 1 defense player per offensive player. We allow more of a distance between the pairs of players as well.
- 2_3 - Just like 2-3 defense in a basketball game. Defensive players each get their own box to maintain and when an opposing player enters, they trail to defend against them. When overlapping happens, we still only allow 1 defensive player to any 1 offensive player (the closest one).




How to Run Plan Guided

We first set the device we want to use, and specify the dataset, logbase, and if we want to, load a specific instance of a trained model (if we dont want the most updated model if performance worsened and such). See the following example.

CUDA_VISIBLE_DEVICES=6 python ./scripts/plan_guided.py --dataset basketball_single_game_wd_act --logbase /local2/dmreynos/diffuser_bball/logs/ --diffusion_epoch epoch_50


How to Run Training 

We run similarly here but we specify parameters also, these can be found in locomotion.py. Following is an example.

CUDA_VISIBLE_DEVICES=2 python scripts/train.py --dataset basketball_single_game_wd --n_diffusion_steps 30 --action_weight 30 --ema_decay 0.996 --learning_rate 0.002 --savepath act30lre3 > "trainingoutput_act30lre3.log" 2>&1 


Generate gifs
Run the given gif pipeline and specify the path of the npy file generated from the previous plan guided.

python full_visual_pipeline.py --path /local2/dmreynos/diffuser_bball/logs/"guided_samples_test_cond100_0.1"/2016.NBA.Raw.SportVU.Game.Logs12.05.2015.POR.at.MIN_dir-1-guided-245K.npy

Generate still vizualization
This one is a little different, but we just need the path after logs/ and without the .npy at the end. The numbers following represent the selection of the 5 trails for a given possession, start frame, end frame, and shooter player number respectively.

python NBA-Player-Movements/shooter_png_dir/visual_2d.py guided_samplesact_(2_3)_50100_0.1/2016.NBA.Raw.SportVU.Game.Logs12.05.2015.POR.at.MIN_dir-15-guided-245K -1 233 300 3
