SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

sudo apt-get install -y libatlas-base-dev libhdf5-dev libhdf5-serial-dev libatlas-base-dev libjasper-dev 
curl -sL https://deb.nodesource.com/setup_16.x | sudo -E bash -
sudo apt-get install -y nodejs

# activate legacy camera system (https://forums.raspberrypi.com/viewtopic.php?t=323390)

sudo cp ${SCRIPT_DIR}/teslagofast.service /etc/systemd/system/teslagofast.service