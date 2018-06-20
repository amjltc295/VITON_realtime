IP=$(/sbin/ip -o -4 addr list eth0 | awk '{print $4}' | cut -d/ -f1)
echo "IP Address: $IP"
CUDA_VISIBLE_DEVICES=2,3 gunicorn \
    -b $IP \
    -w 1 --timeout 120 \
    VITON_API_server:app

