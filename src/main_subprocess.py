import subprocess

# for config in ['drawer_eig_0', 'door_eig_0', 'coffee_eig_0', 'lever_eig_0']:
for config in ['drawer_maxdyn_0', 'door_maxdyn_0', 'coffee_maxdyn_0', 'lever_maxdyn_0', 'button_maxdyn_0']:
    print(f"Running config: {config}")
    subprocess.run(
        ["python3.10", "-m", "src.main", "--config", config],
        check=True,
    )