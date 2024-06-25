code for the working paper "Planning with Diffusion for Professional Basketball Player Behavior Synthesis."


Hueristic problem - to only do this for one team, we need to generate trajectories for all teams, then in post update the defending teams position to be that of hueristics, we do this in several different number of batches


Types of hueristics:
- Original - desc
- Loose - desc
- 2_3 - desc




How to Run Plan Guided

CUDA_VISIBLE_DEVICES=6 python ./scripts/plan_guided.py --dataset basketball_single_game_wd_act --logbase /local2/dmreynos/diffuser_bball/logs/ --diffusion_epoch epoch_50


How to Run Training 

CUDA_VISIBLE_DEVICES=2 python scripts/train.py --dataset basketball_single_game_wd --n_diffusion_steps 30 --action_weight 30 --ema_decay 0.996 --learning_rate 0.002 --savepath act30lre3 > "trainingoutput_act30lre3.log" 2>&1 

Generate gifs

python full_visual_pipeline.py --path /local2/dmreynos/diffuser_bball/logs/"guided_samples_test_cond100_0.1"/2016.NBA.Raw.SportVU.Game.Logs12.05.2015.POR.at.MIN_dir-1-guided-245K.npy

Generate still vizualization

python NBA-Player-Movements/shooter_png_dir/visual_2d.py guided_samplesact_(2_3)_50100_0.1/2016.NBA.Raw.SportVU.Game.Logs12.05.2015.POR.at.MIN_dir-15-guided-245K -1 233 300 3
