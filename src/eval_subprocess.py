import subprocess

for i in range(1):
    for env in ['door']: # 'faucet', 'drawer', 'door', 'button'
        for config in [f'configs_final/{env}_{policy}_{i}' for policy in ['eig', 'maxdyn', 'random']]:
            print(f"Running config: {config}")
            subprocess.run(
                ["python3.10", "-m", "src.main", "--config", config],
                check=True,
            )