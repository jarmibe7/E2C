import subprocess

# for i in range(0, 2):
#     config = f"push_eig_{i}"
"""TODO: 
run maxdyn for mcar
pmaze is catastopic collapsing -- check with previous config
run maxdyn for push
"""
to_run = [
    'button_eig', 'button_maxdyn', 'button_random',
    'coffee_eig', 'coffee_maxdyn', 'coffee_random']
for config in to_run:
    print(f"Running config: {config}")
    subprocess.run(
        ["python3.10", "-m", "src.main", "--config", config],
        check=True,
    )