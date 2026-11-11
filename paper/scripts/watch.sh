#!/bin/bash
# usage: bash paper/scripts/watch.sh
# Rebuilds paper/main.pdf whenever main.tex, main.bib, tables/*.tex or a figure changes (checked every 2 s).
# Keep main.pdf open in a viewer that reloads on change (e.g. evince). Uses latexmk if installed, else tectonic.
# Build files go to paper/build/; the log of the last build is paper/build/build.log.
P=$(cd "$(dirname "$0")/.." && pwd); cd "$P"; mkdir -p build
export PATH=$HOME/.local/bin:$PATH
build() {
  if command -v latexmk >/dev/null; then
    latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=build main.tex > build/build.log 2>&1
  elif command -v tectonic >/dev/null; then
    tectonic --outdir build main.tex > build/build.log 2>&1
  else
    echo "no LaTeX compiler found (install latexmk/TeX Live or tectonic)"; exit 1
  fi
  if [ $? -eq 0 ]; then cp build/main.pdf build/main.pdf.tmp && mv build/main.pdf.tmp main.pdf; echo "$(date '+%T') built main.pdf"
  else echo "$(date '+%T') build FAILED, see paper/build/build.log:"; grep -m3 -A2 '^!' build/build.log || tail -5 build/build.log; fi
}
last=""
while true; do
  now=$(stat -c %Y main.tex main.bib tables/*.tex *.png 2>/dev/null | sort -n | tail -1)
  [ "$now" != "$last" ] && { last=$now; build; }
  sleep 2
done
