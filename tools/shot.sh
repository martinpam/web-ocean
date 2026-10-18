#!/bin/zsh
# Render one frame in headless Chrome and save it to shots/<name>.png
#   tools/shot.sh <name.png> "<extra query params>"
# Needs the dev server running on :8080 (npm start).
name=$1; extra=$2
rm -f "shots/$name"
prof=$(mktemp -d)
"/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" --headless=new --user-data-dir="$prof" \
  --enable-unsafe-webgpu --enable-features=Vulkan --use-angle=metal --ignore-gpu-blocklist \
  --window-size=1280,720 --no-first-run --disable-background-timer-throttling \
  "http://localhost:8080/?w=1280&h=720&ui=0&shot=$name&$extra" >/dev/null 2>&1 &
pid=$!
for i in {1..120}; do [[ -f "shots/$name" ]] && break; sleep 0.5; done
sleep 0.3; kill $pid 2>/dev/null; rm -rf "$prof"
[[ -f "shots/$name" ]] && echo "ok shots/$name" || echo "FAILED $name"
