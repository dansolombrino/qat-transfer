#!/bin/bash
# ssh wrapper with retry/backoff for multi-rig dispatch.
#
# WHY: behemoth's sshd runs the default `MaxSessions 10` and we have no sudo to
# raise it. With 4 vision feeders + 4 text feeders + a gate loop + a watchdog we
# sit exactly at the cap, and any extra call gets "Session open refused by peer".
# ssh then returns non-zero, which dispatchers mis-read as "the job failed" --
# even when the remote work had already completed.
#
# Connection multiplexing does NOT solve this: MaxSessions counts channels per
# connection, so funnelling everything through one master hits the same limit.
# The durable fix is to treat a refusal as a transient condition and retry.
#
# Usage: rssh.sh <host> <command string>
# Exit codes: the remote command's, or 255 if every attempt was refused.
set -u
HOST=$1; shift
CMD="$*"
MAX=6
DELAY=5
for i in $(seq 1 $MAX); do
  ERR=$(mktemp)
  ssh -o BatchMode=yes -o ConnectTimeout=15 "$HOST" "$CMD" 2>"$ERR"
  RC=$?
  if [ $RC -eq 0 ]; then rm -f "$ERR"; exit 0; fi
  # only retry on transport-level refusals, never on genuine remote failures
  if grep -qiE 'session request failed|Session open refused|Connection closed|Connection reset|Connection timed out|kex_exchange|Too many' "$ERR"; then
    cat "$ERR" >&2
    rm -f "$ERR"
    sleep $((DELAY * i))
    continue
  fi
  cat "$ERR" >&2; rm -f "$ERR"
  exit $RC
done
echo "rssh: giving up on $HOST after $MAX refused attempts" >&2
exit 255
