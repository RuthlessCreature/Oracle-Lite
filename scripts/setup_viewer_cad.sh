#!/usr/bin/env bash
set -euo pipefail
APP_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
WORKSPACE_DIR=$(dirname "$APP_DIR")
CAD_SOURCE="$WORKSPACE_DIR/third_party/parasolid-kit"
CAD_PYTHON="$APP_DIR/.venv-cad/bin/python"
CAD_COMMIT=2bc959687c8ca87145c4e0594828c3fa11aab912
mkdir -p "$WORKSPACE_DIR/third_party/parasolid-schemas"
if [[ ! -d "$CAD_SOURCE/.git" ]]; then
  git clone https://github.com/monozukuri-ai/parasolid-kit.git "$CAD_SOURCE"
  git -C "$CAD_SOURCE" checkout "$CAD_COMMIT"
fi
[[ $(git -C "$CAD_SOURCE" rev-parse HEAD) == "$CAD_COMMIT" ]] || { echo 'Unexpected CAD source revision'; exit 1; }
if git -C "$CAD_SOURCE" apply --check "$APP_DIR/scripts/parasolid-kit-oracle.patch"; then
  git -C "$CAD_SOURCE" apply "$APP_DIR/scripts/parasolid-kit-oracle.patch"
else
  git -C "$CAD_SOURCE" apply --reverse --check "$APP_DIR/scripts/parasolid-kit-oracle.patch"
fi
SCHEMA="$WORKSPACE_DIR/third_party/parasolid-schemas/sch_13006.sch_txt"
if [[ ! -f "$SCHEMA" ]]; then
  curl -fsSL https://raw.githubusercontent.com/khoanguyen-3fc/ps-parser/main/assets/sch_13006.s_t -o "$SCHEMA"
  curl -fsSL https://raw.githubusercontent.com/khoanguyen-3fc/ps-parser/main/LICENSE -o "$(dirname "$SCHEMA")/SOURCE-LICENSE.txt"
fi
printf '%s  %s\n' 0dd291ea706fc306f16a78140e05e595e75c85ab63e4077e68941b127fcfdcc3 "$SCHEMA" | sha256sum -c -
# Exact catalog for standard SCH_*_20000 exports. Retain the original bytes
# and the archived source's copyright notice separately from the MIT 13006 file.
SCHEMA_20000="$WORKSPACE_DIR/third_party/parasolid-schemas/sch_20000.sch_txt"
SCHEMA_20000_SOURCE=https://salsa.debian.org/science-team/gmsh/-/raw/c49de80e94234f72219b0e0c9fdcbacbd3ffdf5c
if [[ ! -f "$SCHEMA_20000" ]]; then
  SCHEMA_20000_TEMP=$(mktemp "${SCHEMA_20000}.XXXXXX")
  curl -fsSL "$SCHEMA_20000_SOURCE/contrib/Parasolid/interface_parasolid/schema/sch_20000.sch_txt" -o "$SCHEMA_20000_TEMP"
  printf '%s  %s\n' f66857fb80ce2c85669f0fbd0ba57244cf58b3266da1fae41821185fb66dc730 "$SCHEMA_20000_TEMP" | sha256sum -c -
  mv "$SCHEMA_20000_TEMP" "$SCHEMA_20000"
fi
printf '%s  %s\n' f66857fb80ce2c85669f0fbd0ba57244cf58b3266da1fae41821185fb66dc730 "$SCHEMA_20000" | sha256sum -c -
if [[ ! -f "$(dirname "$SCHEMA_20000")/sch_20000-source-copyright.txt" ]]; then
  curl -fsSL "$SCHEMA_20000_SOURCE/debian/copyright" -o "$(dirname "$SCHEMA_20000")/sch_20000-source-copyright.txt"
fi
if [[ ! -x "$CAD_PYTHON" ]]; then
  "$APP_DIR/.venv/bin/python" -m venv "$APP_DIR/.venv-cad"
fi
export RUSTUP_HOME="$WORKSPACE_DIR/third_party/rust/rustup"
export CARGO_HOME="$WORKSPACE_DIR/third_party/rust/cargo"
export PATH="$CARGO_HOME/bin:$PATH"
if [[ ! -x "$CARGO_HOME/bin/rustc" ]]; then
  curl -fsSL https://sh.rustup.rs -o "$WORKSPACE_DIR/third_party/rustup-init.sh"
  sh "$WORKSPACE_DIR/third_party/rustup-init.sh" -y --profile minimal --default-toolchain 1.99.0 --no-modify-path
fi
"$CAD_PYTHON" -m pip install maturin==1.15.0 cadquery-ocp-novtk==7.9.3.1.1 mapbox-earcut==2.1.0 numpy==2.2.6
(cd "$CAD_SOURCE" && "$APP_DIR/.venv-cad/bin/maturin" build --release --interpreter "$CAD_PYTHON" --out target/oracle-wheels)
"$CAD_PYTHON" -m pip install --no-deps --force-reinstall "$CAD_SOURCE"/target/oracle-wheels/*.whl
printf 'Local CAD display decoder installed in %s\n' "$APP_DIR/.venv-cad"
