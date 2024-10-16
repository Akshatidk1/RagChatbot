py -3 -m pip install fastapi
py -3 -m pip install "uvicorn[standard]"
py -3  run.py

# python installed version
3.11.3

# virtual envirenment setup for windows
virtualenv venv
 
# virtual envirement setup for linux 
python3.11 -m venv venv
source venv/bin/activate

# for installing Virtualenv
sudo apt install python3-virtualenv

# install requirement
pip3.11 install -r requirements.txt

# remove venv
sudo rm -rf venv/

# project run command
python3.11 run.py

# for Starting uvicorn

uvicorn main:app --reload 

# for starting streamlit

streamlit run frontend/interface.py