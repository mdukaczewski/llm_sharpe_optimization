#!/bin/bash
# convert newly completed tickers (all 11 year files present) and run the ON book on them
cd /home/trumpet/strat20
export PYTHONPATH=/home/trumpet/strat20
PY=/home/trumpet/anaconda3/envs/py38/bin/python
while true; do
  done_t=""
  for t in $(ls data/minute/*.json.gz | xargs -n1 basename | cut -d_ -f1 | sort -u); do
    n=$(ls data/minute/${t}_*.json.gz | wc -l)
    if [ "$n" -ge 11 ] && [ ! -f data/minute_npz/$t.npz ]; then done_t="$done_t,$t"; fi
  done
  if [ -n "$done_t" ]; then
    $PY agents/intraday/convert.py ${done_t#,} >> logs/watch_convert.log 2>&1
  fi
  if ! pgrep -f "[f]etch_minute.py" > /dev/null; then echo "fetch finished $(date)" >> logs/watch_convert.log; break; fi
  sleep 60
done
