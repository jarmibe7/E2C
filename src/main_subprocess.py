import subprocess

for i in range(2):
    for env in ['faucet']: # 'coffee', 'button', 'drawer', 'door'
        for config in [f'configs_final/{env}_{policy}_{i}' for policy in ['eig', 'maxdyn', 'random']]:
            print(f"Running config: {config}")
            subprocess.run(
                ["python3.10", "-m", "src.main", "--config", config],
                check=True,
            )