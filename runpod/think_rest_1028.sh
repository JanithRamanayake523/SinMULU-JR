#!/usr/bin/env bash
# after the subset thinking run: reuse its 300 reasoning texts, generate the other 1,938 test questions
cd /workspace/SinMULU-JR
while pgrep -f "sinmmlu.evaluate" >/dev/null || pgrep -f think_1028.sh >/dev/null; do sleep 20; done
S=outputs/qwen35_4b_lora_combined__none__cot_en_role_think__chaton__p1__c0__trnone__mt4096__test_subset300
T=outputs/qwen35_4b_lora_combined__none__cot_en_role_think__chaton__p1__c0__trnone__mt4096__test
[ -f $S/predictions.zip ] || { echo "subset run did not finish"; exit 1; }
mkdir -p $T; [ -f $T/generations.jsonl ] || cp $S/generations.jsonl $T/generations.jsonl
wc -l $T/generations.jsonl
bash runpod/think_1028.sh test
