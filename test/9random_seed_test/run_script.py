import subprocess

# subprocess.run("python CRISPR_IP.py 0")
# subprocess.run("python CRISPR_IP.py 10")
# subprocess.run("python CRISPR_IP.py 20")
# subprocess.run("python CRISPR_IP.py 30")
# subprocess.run("python CRISPR_IP.py 40")
# subprocess.run("python CRISPR_IP.py 50")
# subprocess.run("python CRISPR_IP.py 60")
# subprocess.run("python CRISPR_IP.py 70")
# subprocess.run("python CRISPR_IP.py 80")
# subprocess.run("python CRISPR_IP.py 90")
# subprocess.run("python CRISPR_IP.py 100")
# subprocess.run("python CRISPR_IP.py 110")
# subprocess.run("python CRISPR_IP.py 120")
# subprocess.run("python CRISPR_IP.py 130")
# subprocess.run("python CRISPR_IP.py 140")
# subprocess.run("python CRISPR_IP.py 150")
# subprocess.run("python CRISPR_IP.py 160")
# subprocess.run("python CRISPR_IP.py 170")
# subprocess.run("python CRISPR_IP.py 180")
# subprocess.run("python CRISPR_IP.py 190")

# for i in range(0, 200, 10):
#     subprocess.run(f"python R-CRISPR.py {i}")

# for i in range(0, 200, 10):
#     subprocess.run(f"python CRISPR_M_mismatch_test.py {i}")

# for i in range(0, 200, 10):
#     subprocess.run(f"python R-CRISPR_mismatch.py {i}")

# for i in range(0, 200, 10):
#     subprocess.run(f"python deepcrispr_mismatch.py {i}")

# for i in range(0, 200, 10):
#     subprocess.run(f"python CRISPR-Net_mismatch.py {i}")

# for i in range(0, 200, 10):
#     subprocess.run(f"python CRISPR_IP_mismatch.py {i}")

# for i in range(0, 200, 10):
#     subprocess.run(f"python cnn_std_keras_mismatch.py {i}")

for i in range(0, 200, 10):
    subprocess.run(f"python cfd-score-calculator.py {i}")
