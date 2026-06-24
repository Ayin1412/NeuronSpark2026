#!/bin/bash

nohup vllm serve --config ./vllm.yaml > ./logs/vllm_multi.log 2>&1 &