## Build the image:

docker build -t my-elasticsearch .


## Run the container:

docker run -it \
    --rm \
    --name elasticsearch \
    -m 4GB \
    -p 9200:9200 \
    -p 9300:9300 \
    my-elasticsearch
