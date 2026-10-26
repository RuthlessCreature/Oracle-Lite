# Oracle model backend separation

Talker and Viewer now keep application behavior separate from the inference
server. `TalkerModel` continues to own retrieval, prompts, attachments,
engineering checks and answer continuation. `OpenAICompatibleBackend` owns
endpoint health, authentication, image encoding, request dialect and response
decoding. The same application path can target Strata or a self-hosted vLLM
endpoint without putting vendor-specific request fields in marking and
measurement code.

## Active profile

The Viewer and Talker use the already-installed Qwen3.8-27B vLLM service.
The local app connects to `127.0.0.1:59000`; a user systemd SSH tunnel forwards
that loopback port to the existing cloud vLLM loopback endpoint. The cloud
inference port is not exposed publicly. The dedicated SSH key is restricted to
forwarding that one port and cannot open a shell. The API key is kept in a
user-only environment file outside the repository.

As of 2026-10-23, `oracle-qwen-inference` serves the existing checkpoint on GPU 1
and `oracle-qwen-trainer` runs a separate QLoRA adapter on GPU 0. The former
two-GPU container `qwen38-vllm` is retained, stopped, with its configuration
unchanged for rollback. Model files remain mounted read-only. No second model
weights were downloaded. The isolated inference instance uses a 65,536-token
context rather than the former instance's 262,144 tokens; callers must retain
complete evidence within that limit or split tasks. It serves the same private
loopback endpoint, so Viewer and Talker retain their existing tunnel profile.

The active service profile is in
`~/.config/systemd/user/oracle-viewer.service.d/model-backend.conf`; the
credential is in `~/.config/oracle-viewer/qwen-vllm.env`. These local runtime
files are intentionally outside the project repository. The request adapter
is in `talker/inference_backend.py`.

## Cloud adapter training

Changing an inference endpoint alone does not train weights. The separate cloud
job now trains rank-8 QLoRA on MSOP page images plus native text with complete
coordinate labels. There are 75 training pages / 1,211 printed points from
613-13599 and 806-69194; 73 pages / 579 points from 613-16076 form the independent
project validation set. 817-07781 is excluded. A conflicting 806 page is excluded
rather than choosing an unverified coordinate variant. Prompt/image tokens are
masked from loss; complete answers are never truncated. Existing XML files and
model-generated FAI placements are not input labels.

The first attempt exhausted activation/logit memory at step 8. The retry uses
only the logits that predict unmasked answer tokens and checkpoints cross
entropy chunks. A unit test verifies equal loss and parameter gradients against
ordinary masked causal SFT. This avoids dropping evidence or shrinking point
groups. Intermediate adapters/optimizer state are saved every two steps.

This job trains source coordinate reading only. It does not establish correct
CAD registration, station planning, optical measurement accuracy, or XML output.
The intended final workflow remains 2D/3D/MSOP → FAI registration → measurement
setups → virtual shots and pixel/world correspondences → measurement-program
XML. XML is an output artifact, never project evidence fed back to manufacture
an apparently successful plan.

Cloud files are under `/home/ubuntu/wjkj/oracle-training`; the original base
checkpoint is `/home/ubuntu/wjkj/models/Qwen3.8-27B`. The adapter and logs are in
`runs/source-reading-20261023`. Training and paired base/adapter validation run
outside the Viewer process, survive a Viewer restart, and do not automatically
deploy a candidate. The progress mirror user service is
`oracle-cloud-training-monitor.service`; `/api/viewer/cloud-training` and the
Viewer progress page report actual receipts and optimizer steps separately from
FAI coverage.

The completed adapter is registered under a separate candidate alias in vLLM;
the default Viewer alias still addresses the frozen base. Paired validation uses
that same BF16 serving runtime, the same source image/text and the same output
schema for base and adapter. Candidate registration is not deployment. The
earlier Transformers/4-bit validation process was superseded by this deployment
runtime comparison, and its cancellation receipt remains available.

The completed 2026-10-23 run has 19 optimizer steps, 58,363,904 trainable adapter
parameters and a saved 223 MiB safetensors artifact. Paired held-out evaluation
completed all 73 pages and 579 points for each model with no request errors or
truncated final answers. Both models reproduced all coordinate content. Raw
JSON compliance was 0/73 for the base (Markdown-wrapped JSON) and 73/73 for the
adapter. This demonstrates output-format adaptation, not improved geometric
registration or improved coordinate content. The source adapter remains a
separate candidate; Viewer still addresses the base alias. The dedicated
GPU-0 evaluation server is stopped after validation, leaving GPU 0 free.

The API status reports the active `inference_backend` and `model_alias`. The
marking fingerprint includes both, so an inference model change produces a new
run identity instead of silently reusing the prior model's run receipt.
