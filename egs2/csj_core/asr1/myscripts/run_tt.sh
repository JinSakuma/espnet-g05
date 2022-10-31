#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set=train_sp
valid_set=valid
test_sets=test

./asr.sh \
    --ngpu 2 \
    --stage 7 \
    --stop_stage 100 \
    --lang jp \
    --use_lm false \
    --use_word_lm false \
    --token_type char \
    --asr_config myconf/train_asr_tt.yaml \
    --inference_config myconf/decode_tt.yaml \
    --inference_asr_model valid.loss.ave.pth \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --lm_train_text "data/train/text" "$@"
