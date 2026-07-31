python3 -u ppo.py --env_id="PickAnything-v1" --num_envs=2048 --update_epochs=8 --num_minibatches=32 --total_timesteps=2_000_000 --eval_freq=10 --num-steps=20 --object-sources cube ycb
