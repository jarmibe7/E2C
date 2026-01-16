import subprocess

for i in range(9, 11):
    config = f"push_active_{i}"
    print(f"Running config: {config}")
    subprocess.run(
        ["python3.10", "-m", "src.main", "--config", config],
        check=True,
    )