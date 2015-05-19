git clone https://rfvander@xstack.exascale-tech.com/gerrit/xstack
cd xstack/ocr/runtime/ocr-x86
./install.sh
export OCR_INSTALL=${PWD}/ocr-install
export LD_LIBRARY_PATH=${OCR_INSTALL}/lib:${LD_LIBRARY_PATH}
