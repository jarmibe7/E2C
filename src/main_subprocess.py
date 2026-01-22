import subprocess

# for i in range(0, 2):
#     config = f"push_eig_{i}"
"""TODO: 
run maxdyn for mcar
pmaze is catastopic collapsing -- check with previous config
run maxdyn for push
"""
for config in ['push_eig_2']:
    print(f"Running config: {config}")
    subprocess.run(
        ["python3.10", "-m", "src.main", "--config", config],
        check=True,
    )