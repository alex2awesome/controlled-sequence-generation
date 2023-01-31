DEFAULT_REPO='git+https://bbgithub.dev.bloomberg.com/aspangher/controlled-sequence-gen.git'
DEFAULT_BRANCH='master'
DEFAULT_PACKAGE=$DEFAULT_REPO@$DEFAULT_BRANCH

DEFAULT_JOB_SIZE='Custom'
#DEFAULT_FRAMEWORK='python-3.7-rhel-cuda-10.2'
DEFAULT_FRAMEWORK='python-3.8-rhel-cuda-11.1'
DEFAULT_GIT_IDENTIY='spectro-oauth-aspangher'
DEFAULT_BCS_IDENTITY='aspangher-cluster-test'
ENV=bb

## gpus
num_nodes=1
num_gpus_per_node=1
if [[ $num_nodes -gt 1 ]]
then
  APPROACH='distributed-pytorch'
  worker_args="--node-num-gpus $num_gpus_per_node --num-workers $num_nodes --node-num-cores 4 --node-memory 60G"
else
  APPROACH='single'
  worker_args="--node-num-gpus $num_gpus_per_node --node-num-cores 4 --node-memory 60G"
fi

katie compute run \
        $APPROACH \
        --compute-framework $DEFAULT_FRAMEWORK \
        --node-size $DEFAULT_JOB_SIZE \
        $worker_args \
        --python-module editing.run_stage_one \
        --identities bcs=$DEFAULT_BCS_IDENTITY git=$DEFAULT_GIT_IDENTIY \
        --pip-packages $DEFAULT_PACKAGE \
        --env NCCL_ASYNC_ERROR_HANDLING=1 NCCL_LL_THRESHOLD=0 NCCL_DEBUG=INFO env=$ENV \
        -- \
        --results_dir tmp \
        --mask_type grad \
        --grad_type normal_l1 \
        --pretrained_discrim_path "./trial-Sequential, flattened sentences, large-corpus Fine-tuned LM__epoch=07-f1_macro=0.58.ckpt" \
        --pretrained_lm_model_path gpt2-medium-expanded-embeddings \
        --pretrained_editor_lm_path t5-base \
        --real_data_file data/news-discourse-processed.tsv  \
        --remote_editor_upload aspangher/controlled-sequence-gen/./t5-trained-editor-noun-heuristics__l1-norm-grad \
        --spacy_model_file aspangher/en_core_web_lg-2.3.1/ \
        --use_heuristic_masks
