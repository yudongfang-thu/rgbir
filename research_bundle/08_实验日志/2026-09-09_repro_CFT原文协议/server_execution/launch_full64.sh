#!/bin/bash
export RGBIR90_PROJECT_ROOT=/mnt/dataX/ydf/projects/RGBT_campaign_90
/mnt/dataX/ydf/projects/RGBT_campaign_90/environments/cft90/bin/python -B -u /mnt/dataX/ydf/projects/RGBT_campaign_90/tools/project_resource_guard.py --lease-file /mnt/dataX/ydf/projects/RGBT_campaign_90/runs/.project_resource_leases.json run --job-id cft_author_full64 --kind eval --expected-vram-mib 15360 --expected-rss-mib 32768 --free-safety-mib 2048 -- /mnt/dataX/ydf/projects/RGBT_campaign_90/environments/cft90/bin/python -B -u /mnt/dataX/ydf/projects/RGBT_campaign_90/artifacts/cft_author_protocol_20260909_attempt1/run_author_eval.py --mode full --batch 64 --output /mnt/dataX/ydf/projects/RGBT_campaign_90/artifacts/cft_author_protocol_20260909_attempt1/official_test_attempt1 > /mnt/dataX/ydf/projects/RGBT_campaign_90/artifacts/cft_author_protocol_20260909_attempt1/official_test.log 2>&1
rc=$?
printf "%s\n" "$rc" > /mnt/dataX/ydf/projects/RGBT_campaign_90/artifacts/cft_author_protocol_20260909_attempt1/official_test.exit
exit "$rc"
