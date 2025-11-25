#!/bin/bash
source /home/SORBUS-admin/front-query/.vent/bin/activate
cd /home/SORBUS-admin/front-query/Program3
exec uvicorn main:app --host 0.0.0.0 --port 8000
