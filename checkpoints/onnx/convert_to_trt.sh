# convert all onnx files to trt files
# link to nvinfer libs
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:../../third_party/TensorRT/build/out
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:../../third_party/TensorRT-10.2.0.19/targets/x86_64-linux-gnu/lib

# iterate over all onnx files in the checkpoints/onnx directory
for file in $(ls ./*.onnx); do
    ../../third_party/TensorRT/build/out/trtexec --onnx=$file --saveEngine=../trt/${file%.onnx}.engine --verbose --fp16
done

