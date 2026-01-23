import subprocess

# for i in range(0, 2):
#     config = f"push_eig_{i}"
"""TODO: 
run maxdyn for mcar
pmaze is catastopic collapsing -- check with previous config
run maxdyn for push
"""
for config in ['plate_eig_0', 'plate_random_0', 'push_eig_2']:
    # 'button_random_0', 'plate_maxdyn_0','button_eig_0']: 
    print(f"Running config: {config}")
    subprocess.run(
        ["python3.10", "-m", "src.main", "--config", config],
        check=True,
    )