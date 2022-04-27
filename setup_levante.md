# setup for `levante.dkrz.de`

For `mistral.dkrz.de` follow the same guide.

On `levante` [clone](https://docs.github.com/en/repositories/creating-and-managing-repositories/cloning-a-repository) the [github](https://en.wikipedia.org/wiki/GitHub) [repository](https://docs.github.com/en/repositories):
```
git clone https://github.com/observingClouds/xbitinfo.git
cd xbitinfo
```

Create a new [conda environment](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-with-commands) and register the python [kernel](https://docs.dkrz.de/doc/software&services/jupyterhub/kernels.html#use-your-own-kernel):
```
module load python3
mamba env create -f environment.yml  # use conda on mistral
source activate bitinfo
python -m ipykernel install --user --name bitinfo --display-name=bitinfo
```

Create `~/jupyter_preload` on the supercomputer:
```
source activate bitinfo
```

Get [jupyter](https://docs.jupyter.org/en/latest/) with [`start-jupyter script from DKRZ`](https://gitlab.dkrz.de/k204213/ssh_scripts/-/blob/master/start-jupyter) on your laptop:
```
wget https://gitlab.dkrz.de/k204213/ssh_scripts/-/raw/master/start-jupyter
```

Personalize the following lines:
- [L59](https://gitlab.dkrz.de/k204213/ssh_scripts/-/blob/master/start-jupyter#L59): `SJ_ACCTCODE=mh0727` with your account number
- [L65](https://gitlab.dkrz.de/k204213/ssh_scripts/-/blob/master/start-jupyter#L65): `SJ_USERNAME=m300524` with your username
- [L70](https://gitlab.dkrz.de/k204213/ssh_scripts/-/blob/master/start-jupyter#L70): `SJ_COMMAND=lab` for using jupyter `notebook` or jupyter `lab`
- [L90](https://gitlab.dkrz.de/k204213/ssh_scripts/-/blob/master/start-jupyter#L90): `SJ_INCFILE="jupyter_preload"`
- [L96](https://gitlab.dkrz.de/k204213/ssh_scripts/-/blob/master/start-jupyter#L96): `SJ_FRONTEND_HOST="levante.dkrz.de"` (keep as is for `mistral`)

Now `start-jupyter` (overwriting script options is possible in the command line, see
[available options](https://gitlab.dkrz.de/k204213/ssh_scripts/-/blob/master/start-jupyter#L141)):
`sh start-jupyter`
