echo "BEGIN: ISMgas environment setup"  && \

conda install python==3.9  && \
pip install -e .  && \
pip install scipy  && \
pip install matplotlib  && \
pip install ipykernel  && \
pip install astropy  && \
pip install requests  && \
pip install sparclclient   && \
pip install pandas  && \
pip install astroscrappy  && \
pip install lmfit &&\
pip install symfit &&\
pip install regions &&\
pip install beautifyPlot &&\
echo "END: ISMgas environment setup complete" 
