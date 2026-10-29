
apt-get update
apt-get install -y software-properties-common python3-pip wget tcpdump


add-apt-repository -y ppa:wireshark-dev/stable
DEBIAN_FRONTEND=noninteractive apt-get install -y tshark


wget https://dl.google.com/linux/direct/google-chrome-stable_current_amd64.deb
DEBIAN_FRONTEND=noninteractive apt-get install -y ./google-chrome-stable_current_amd64.deb
rm ./google-chrome-stable_current_amd64.deb 


pip3 install pyshark selenium nest_asyncio