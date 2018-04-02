CUDA_VISIBLE_DEVICES=2,3 gunicorn \
    -b 140.112.29.182 \
    -w 1 --timeout 120 \
    VITON_API_server:app

