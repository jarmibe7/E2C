import subprocess

for i in range(2):
    # for config in [f'configs_final/{env}_{policy}_{i}' for policy in ['eig', 'maxdyn', 'random']]:
    for policy in ['eig', 'maxdyn', 'random']:
        for env in ['door', 'drawer', 'faucet']: # 'button', 'coffee'
            config = f'configs_change_cam/{env}_{policy}_{i}'
            print(f"Running config: {config}")
            subprocess.run(
                ["python3.10", "-m", "src.main", "--config", config],
                check=True,
            )