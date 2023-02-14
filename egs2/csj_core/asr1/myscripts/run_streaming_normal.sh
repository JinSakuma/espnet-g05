#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

#stage=12
skip_data_prep=false
#train_set=train_sp
train_set=train
valid_set=valid
test_sets=test

#asr_config=myconf/train_asr_streaming_conformer_v2.yaml
#asr_config=myconf/train_asr_streaming_conformer_blk16_hop4_la0.yaml
asr_config=myconf/train_asr_streaming3_conformer_blk16_hop4_la4_offset100.yaml
#asr_config=myconf/train_asr_streaming2_conformer_blk16_hop4_la4_offset100.yaml
inference_config=myconf/decode_asr_streaming_normal_chunk512.yaml
#inference_config=myconf/decode_asr_streaming_m1.yaml
#inference_config=conf/decode_asr_streaming.yaml
bpe_train_text=dump/raw/train_sp/text
lm_config=myconf/train_lm_offset100.yaml
#lm_config=myconf/train_lm_offset100_preprocessed.yaml
use_lm=true
use_wordlm=false

# speed perturbation related
# (train_set will be "${train_set}_sp" if speed_perturb_factors is specified)
speed_perturb_factors="0.9 1.0 1.1"

./asr.sh                                               \
    --ngpu 4                                           \
    --stage 12                                         \
    --stop_stage 100                                   \
    --skip_data_prep "${skip_data_prep}"               \
    --use_dual_delay false                             \
    --use_streaming true                               \
    --lang jp                                          \
    --audio_format wav                                 \
    --feats_type raw                                   \
    --token_type char                                  \
    --use_eou false                                    \
    --use_lm ${use_lm}                                 \
    --use_word_lm ${use_wordlm}                        \
    --lm_config "${lm_config}"                         \
    --asr_config "${asr_config}"                       \
    --inference_config "${inference_config}"           \
    --train_set "${train_set}"                         \
    --valid_set "${valid_set}"                         \
    --test_sets "${test_sets}"                         \
    --speed_perturb_factors "${speed_perturb_factors}" \
    --lm_train_text "data/${train_set}/text" "$@"
    #--gpu_inference true                               \
    #--asr_speech_fold_length 512 \
    #--asr_text_fold_length 150 \
    #--lm_fold_length 150 \
    #--lm_train_text "data/${train_set}/text" "$@"
    #--nbpe 300                                         \
    #--bpe_train_text ${bpe_train_text}        \