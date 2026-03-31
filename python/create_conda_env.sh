# create conda environment
# if roboLab environment already exists, remove it
if conda env list | grep -q "roboLab"; then
    conda remove -n roboLab -y
fi
conda create -n roboLab python=3.10 -y

conda activate roboLab
pip install eclipse-zenoh
pip install "protobuf>=3.20.0,<4.0.0"

python -m pip install   torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1   --index-url https://download.pytorch.org/whl/cu118
cd ./third_party/Grounded-SAM-2
pip install -e . -v
pip install --no-build-isolation -e grounding_dino -v
pip install opencv-python supervision addict pycocotools yapf timm transformers
pip install 'httpx[socks]' socksio

# if bert is not under the cache directory, download it
if ! ls ~/.cache/transformers/models--bert-base-uncased > /dev/null 2>&1; then
    python -c "from transformers import AutoTokenizer, BertModel; AutoTokenizer.from_pretrained('bert-base-uncased', cache_dir='~/.cache/transformers'); BertModel.from_pretrained('bert-base-uncased', cache_dir='~/.cache/transformers')"
fi

cd ../../
