#!/bin/bash
export RGBIR90_PROJECT_ROOT=/mnt/dataX/ydf/projects/RGBT_campaign_90
/mnt/dataX/ydf/projects/RGBT_campaign_90/environments/cft90/bin/python -B -u /mnt/dataX/ydf/projects/RGBT_campaign_90/tools/project_resource_guard.py --lease-file /mnt/dataX/ydf/projects/RGBT_campaign_90/runs/.project_resource_leases.json run --job-id cft_author_train_canary32_cap68 --kind train --expected-vram-mib 17152 --expected-rss-mib 65536 --free-safety-mib 2048 -- /mnt/dataX/ydf/projects/RGBT_campaign_90/environments/cft90/bin/python -B -u /mnt/dataX/ydf/projects/RGBT_campaign_90/artifacts/cft_author_protocol_20260909_attempt1/train_canary_wrapper.py --weights /mnt/dataX/ydf/projects/RGBT_campaign_90/external_reproductions/cft/author_initialization/yolov5l.pt --memory-fraction 0.68 --output /mnt/dataX/ydf/projects/RGBT_campaign_90/artifacts/cft_author_protocol_20260909_attempt1/train_canary32_attempt2 > /mnt/dataX/ydf/projects/RGBT_campaign_90/artifacts/cft_author_protocol_20260909_attempt1/train_canary32_cap68.log 2>&1
rc=$?
printf "%s\n" "$rc" > /mnt/dataX/ydf/projects/RGBT_campaign_90/artifacts/cft_author_protocol_20260909_attempt1/train_canary32_cap68.exit
exit "$rc"
