import subprocess

for i in range(0, 3):
    config = f"mcar_active_{i}"
# for config in ['pointmaze_random_2', 'push_active_24']:
    print(f"Running config: {config}")
    subprocess.run(
        ["python3.10", "-m", "src.main", "--config", config],
        check=True,
    )