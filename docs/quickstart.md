Quickstart:

1. Export ONNX:
   python3 src/export_onnx.py --checkpoint sample.pt --prefill prefill.onnx --decode decode.onnx

2. Build TRT engine:
   python3 src/build_engine.py --onnx prefill.onnx --engine prefill.engine

3. Prepare Triton model repo (mount to container /models):
   Copy triton_model_repo to host and mount.

4. Start Triton:
   docker run --gpus all -v $(pwd)/triton_model_repo:/models nvcr.io/nvidia/tritonserver:23.11      tritonserver --model-repository=/models --strict-model-config=false

5. Run bench:
   bash bench/run_triton_perf.sh
