#!/usr/bin/env bash
# Fetch an AI2-THOR Unity build with parallel range requests (single connections to S3 crawl from here),
# verify the published sha256, and unpack it where ai2thor looks (~/.ai2thor/releases/<name>), so the
# Controller skips its own download.
#   bash scripts/fetch_thor_build.sh [CloudRendering|Linux64] [commit]
set -eo pipefail
PLATFORM="${1:-CloudRendering}"
COMMIT="${2:-f0825767cd50d69f666c7f282e54abfe58f1e917}"   # ai2thor 5.0.0 default build
PARTS="${PARTS:-8}"
NAME="thor-${PLATFORM}-${COMMIT}"
URL="http://s3-us-west-2.amazonaws.com/ai2-thor-public/builds/${NAME}.zip"
REL="$HOME/.ai2thor/releases"; TMP="$HOME/.ai2thor/tmp"
[ -d "$REL/$NAME" ] && { echo "$NAME already installed"; exit 0; }
mkdir -p "$REL" "$TMP"
work="$(mktemp -d "$TMP/fetch.XXXX")"
trap 'rm -rf "$work"' EXIT

SHA="$(curl -fsSL "${URL%.zip}.sha256" | tr -d '[:space:]')"
SIZE="$(curl -fsSIL "$URL" | awk 'tolower($1)=="content-length:"{print $2}' | tr -d '\r' | tail -1)"
echo "$NAME: $((SIZE / 1048576)) MB, $PARTS parts"
chunk=$(( (SIZE + PARTS - 1) / PARTS ))
for i in $(seq 0 $((PARTS - 1))); do
  start=$((i * chunk)); end=$((start + chunk - 1)); [ $end -ge $SIZE ] && end=$((SIZE - 1))
  curl -fsSL --retry 5 --retry-delay 2 -C - -r "$start-$end" -o "$work/part$i" "$URL" &
done
wait
cat $(for i in $(seq 0 $((PARTS - 1))); do echo "$work/part$i"; done) > "$work/build.zip"
echo "$SHA  $work/build.zip" | sha256sum -c -
unzip -q "$work/build.zip" -d "$work/x"
mv "$work/x" "$REL/$NAME"
chmod +x "$REL/$NAME"/thor-* 2>/dev/null || true
echo "installed $REL/$NAME"
