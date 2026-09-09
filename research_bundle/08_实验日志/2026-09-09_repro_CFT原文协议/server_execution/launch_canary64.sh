#!/bin/bash
export RGBIR90_PROJECT_ROOT=/mnt/dataX/ydf/projects/RGBT_campaign_90
/mnt/dataX/ydf/projects/RGBT_campaign_90/environments/cft90/bin/python -B -u /mnt/dataX/ydf/projects/RGBT_campaign_90/tools/project_resource_guard.py --lease-file /mnt/dataX/ydf/projects/RGBT_campaign_90/runs/.project_resource_leases.json run --job-id cft_author_canary64 --kind eval --expected-vram-mib 16384 --expected-rss-mib 65536 --free-safety-mib 2048 -- /mnt/dataX/ydf/projects/RGBT_campaign_90/environments/cft90/bin/python -B -u /mnt/dataX/ydf/projects/RGBT_campaign_90/artifacts/cft_author_protocol_20260909_attempt1/run_author_eval.py --mode canary --batch 64 --output /mnt/dataX/ydf/projects/RGBT_campaign_90/artifacts/cft_author_protocol_20260909_attempt1/canary64_attempt1 > /mnt/dataX/ydf/projects/RGBT_campaign_90/artifacts/cft_author_protocol_20260909_attempt1/canary64.log 2>&1
rc=$?
printf "%s\n" "$rc" > /mnt/dataX/ydf/projects/RGBT_campaign_90/artifacts/cft_author_protocol_20260909_attempt1/canary64.exit
exit "$rc"
