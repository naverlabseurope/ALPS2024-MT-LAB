# ALPS2022-MT-LAB
Python notebook and models for the [MT Lab @ ALPS 2022](http://lig-alps.imag.fr)

This repository is a modified version of [another tutorial on NMT](https://github.com/nyu-dl/AMMI-2019-NLP-Part2/blob/master/02-day-RLM%26NMT/02.c.NMT/NMT.ipynb) by Kyunghyun Cho et al.

### Running on Google Colab

1. Go to https://colab.research.google.com
2. Under the "GitHub" tab, type the URL of this repo (https://github.com/naverlabseurope/ALPS2022-MT-LAB), then click on "NMT.ipynb"

![colab](https://user-images.githubusercontent.com/1795321/149558712-e71a0148-a340-455d-9dbb-5809f900773c.png)

3. In the Colab menu, go to "Runtime / Change runtime type", then select "GPU" in the "Hardware accelerator" drop-down list

![menu](https://user-images.githubusercontent.com/1795321/149558757-faa37df1-91a6-44d9-ada6-b6538f672b21.png)
![runtime_type](https://user-images.githubusercontent.com/1795321/149558769-34256107-f504-416c-9353-6f61c7835dd1.png)

4. Open [this link](https://drive.google.com/drive/folders/1E07YaKths98YpoBCH2PjdtTPqOXgfdZB?usp=sharing) and connect to your Google Drive account
5. Then go to "Shared with me" in your Google Drive, right-click the "ALPS2022-NMT" folder and select "Add shortcut to Drive"

![drive](https://user-images.githubusercontent.com/1795321/149558193-c7d008e7-09c8-418d-8fcf-2cfb517a52dc.png)

6. Start playing with the notebook

### Running the notebook on your own computer

```
git clone https://github.com/naverlabseurope/ALPS2022-MT-LAB.git
cd ALPS2022-MT-LAB
scripts/setup.sh            # creates a Python environment, installs the dependencies and downloads the data and models
scripts/run-notebook.sh     # starts a jupyter notebook where the lab will take place
```

### Running the notebook remotely

In the following, replace `HOSTNAME` by the hostname of your server, and `USERNAME` by your username.

1. SSH to the server, install the repo and start the notebook
```
ssh HOSTNAME
git clone https://github.com/naverlabseurope/ALPS2022-MT-LAB.git
cd ALPS2022-MT-LAB
scripts/setup.sh
scripts/run-notebook.sh  # modify this script to change the port if 8080 is already used
```

2. Create an SSH tunnel from your machine
```
ssh -L 8080:localhost:22 USERNAME@HOSTNAME
```

3. Open http://HOSTNAME:80880/notebooks/NMT.ipynb in your favorite browser.
4. Enjoy!

### Command line interface for training models
