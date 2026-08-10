#!/usr/bin/env bash

# Normalize an MP4 directory tree to a target presentation frame rate.
# Frames are sampled with FFmpeg's fps filter, so playback duration is kept
# approximately unchanged. Inputs that already match the target FPS are copied.

set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  convert_videos_fps.sh --in-place INPUT_DIR [TARGET_FPS]
  convert_videos_fps.sh INPUT_DIR OUTPUT_DIR [TARGET_FPS]

Examples:
  convert_videos_fps.sh --in-place data/expert_videos 10
  convert_videos_fps.sh data/expert_videos artifacts/expert_videos_10fps 10

In --in-place mode, each output is stored beside its source using a suffix,
for example episode0.mp4 -> episode0_fps10.mp4. Previously generated files
with that suffix are skipped as inputs.

In separate-output mode, the output directory must not be the input directory
or a child of it. Existing output files are overwritten. Audio is omitted
because RoboTwin rollout videos do not contain an audio track.
EOF
}

IN_PLACE=false
if [[ "${1:-}" == "--in-place" ]]; then
  if (( $# < 2 || $# > 3 )); then
    usage >&2
    exit 2
  fi
  IN_PLACE=true
  INPUT_DIR=$2
  OUTPUT_DIR=
  TARGET_FPS=${3:-10}
else
  if (( $# < 2 || $# > 3 )); then
    usage >&2
    exit 2
  fi
  INPUT_DIR=$1
  OUTPUT_DIR=$2
  TARGET_FPS=${3:-10}
fi

if [[ ! -d "$INPUT_DIR" ]]; then
  echo "ERROR: input directory does not exist: $INPUT_DIR" >&2
  exit 2
fi
if [[ ! "$TARGET_FPS" =~ ^[0-9]+([.][0-9]+)?$ ]] || ! awk -v fps="$TARGET_FPS" 'BEGIN { exit !(fps > 0) }'; then
  echo "ERROR: TARGET_FPS must be positive: $TARGET_FPS" >&2
  exit 2
fi
for tool in ffmpeg ffprobe; do
  if ! command -v "$tool" >/dev/null 2>&1; then
    echo "ERROR: $tool is required but is not available" >&2
    exit 2
  fi
done

INPUT_ABS=$(realpath "$INPUT_DIR")
FPS_SUFFIX=${TARGET_FPS//./_}
if [[ "$IN_PLACE" == false ]]; then
  OUTPUT_ABS=$(realpath -m "$OUTPUT_DIR")
  if [[ "$OUTPUT_ABS" == "$INPUT_ABS" || "$OUTPUT_ABS" == "$INPUT_ABS/"* ]]; then
    echo "ERROR: output must be outside the input tree" >&2
    exit 2
  fi
  mkdir -p "$OUTPUT_ABS"
fi

converted=0
copied=0
skipped=0
total=0

while IFS= read -r -d '' src; do
  relative=${src#"$INPUT_ABS"/}
  if [[ "$IN_PLACE" == true ]]; then
    dst="${src%.*}_fps${FPS_SUFFIX}.mp4"
  else
    dst="$OUTPUT_ABS/$relative"
    mkdir -p "$(dirname "$dst")"
  fi

  if ! stream_index=$(ffprobe -v error -select_streams v:0 \
      -show_entries stream=index -of default=noprint_wrappers=1:nokey=1 \
      "$src") || [[ -z "$stream_index" ]]; then
    echo "SKIP    no video stream  $relative"
    ((skipped += 1))
    ((total += 1))
    continue
  fi

  source_rate=$(ffprobe -v error -select_streams v:0 \
    -show_entries stream=avg_frame_rate -of default=noprint_wrappers=1:nokey=1 \
    "$src")

  if awk -v rate="$source_rate" -v target="$TARGET_FPS" 'BEGIN {
      split(rate, parts, "/");
      fps = (parts[2] == 0) ? 0 : parts[1] / parts[2];
      exit !(fps > target - 0.001 && fps < target + 0.001);
    }'; then
    cp -p --reflink=auto "$src" "$dst"
    echo "COPY    $source_rate  $relative"
    ((copied += 1))
  else
    ffmpeg -nostdin -hide_banner -loglevel error -y \
      -i "$src" \
      -map 0:v:0 \
      -vf "fps=${TARGET_FPS}:round=near" \
      -fps_mode cfr \
      -c:v libx264 \
      -preset medium \
      -crf 18 \
      -pix_fmt yuv420p \
      -movflags +faststart \
      -an \
      "$dst"
    output_rate=$(ffprobe -v error -select_streams v:0 \
      -show_entries stream=avg_frame_rate -of default=noprint_wrappers=1:nokey=1 \
      "$dst")
    echo "CONVERT $source_rate -> $output_rate  $relative"
    ((converted += 1))
  fi
  ((total += 1))
done < <(find "$INPUT_ABS" -type f -iname '*.mp4' \
  ! -iname "*_fps${FPS_SUFFIX}.mp4" -print0 | sort -z)

echo "Processed $total video(s): converted=$converted, already_target_fps=$copied, skipped=$skipped"
if [[ "$IN_PLACE" == true ]]; then
  echo "Output: beside source videos with suffix _fps${FPS_SUFFIX}.mp4"
else
  echo "Output: $OUTPUT_ABS"
fi
