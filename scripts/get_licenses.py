import itertools
import webbrowser
import subprocess
import pandas as pd

subprocess.call("touch test.txt")
subprocess.call("grep '\"license\":' C:/Users/358572/Anaconda3/envs/geowq/conda-meta/*.json > test.txt", shell=True)

env = pd.read_csv("environment.yml", header=None)
env = env.iloc[5:len(env)]
env = pd.DataFrame([x.replace("-", "", 1).strip() for x in env[0]], columns=["name"])
not_commented = [not x.__contains__("#") for x in env["name"]]
pkgs = list(itertools.compress([x for x in env["name"]], not_commented))

links_raw = ["https://anaconda.org/conda-forge/" + x for x in pkgs]
links = pd.DataFrame(links_raw, columns=["url"])
links.to_csv("links.csv", index=False)

[webbrowser.open(x) for x in links_raw]
