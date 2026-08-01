# python3 -u ppo.py --env_id="PickCube-v1" --num_envs=2048 --update_epochs=8 --num_minibatches=32 --total_timesteps=10_000_000 --eval_freq=10 --num-steps=20

python3 -u ppo.py --env_id="PickAnything-v1" --num_envs=2048 --update_epochs=8 --num_minibatches=32 --total_timesteps=10_000_000 --eval_freq=10 --num-steps=20 --object-sources cube ycb interndata
python3 -u ppo.py --env_id="PickAnything-v1" --evaluate --checkpoint=runs/PickAnything-v1__ppo__1__1785540304/final_ckpt.pt --num_eval_envs=1 --num-eval-steps=1000
