import subprocess

for i in range(5):
    config = f"push_random_{i}"
    print(f"Running config: {config}")
    subprocess.run(
        ["python3.10", "-m", "src.main", "--config", config],
        check=True,
    )