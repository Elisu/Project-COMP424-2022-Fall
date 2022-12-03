import subprocess


for b in range(6,13):
    subprocess.call("python simulator.py --player_1 student_agent --player_2 random_agent --board_size "+ str(b), shell=True)
