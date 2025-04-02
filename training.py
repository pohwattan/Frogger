import subprocess

def train_agent(restart):
    subprocess.run(f"python3 main.py --player=agent --train=q --screen=medium --steps=500 --restart={restart} --speed=fast", shell=True)

if __name__ == '__main__':

    episodes = [21,20,19,18,19,18,17,16]
    restart = [1, 2, 3, 4, 5, 6, 7, 8]

    for r in restart:
        for _ in range(len(episodes)):
            train_agent(r)
