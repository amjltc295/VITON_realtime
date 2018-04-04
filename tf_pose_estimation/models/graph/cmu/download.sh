#!/bin/bash

echo "download model graph : cmu"

extract_download_url() {

        url=$( wget -q -O - $1 |  grep -o 'http*://download[^"]*' | tail -n 1 )
        echo "$url"

}

wget $( extract_download_url http://www.mediafire.com/file/1pyjsjl0p93x27c/graph_freeze.pb ) -O graph_freeze.pb
wget $( extract_download_url http://www.mediafire.com/file/qlzzr20mpocnpa3/graph_opt.pb ) -O graph_opt.pb
wget $( extract_download_url http://www.mediafire.com/file/i72ll9k5i7x6qfh/graph.pb ) -O graph.pb
