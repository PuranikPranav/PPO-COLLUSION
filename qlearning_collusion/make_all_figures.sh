#!/usr/bin/env bash
# Regenerate every figure and report for one market structure.
#
#   MARKET_CONFIG=three_firm_dist ./qlearning_collusion/make_all_figures.sh
#
# Assumes the runs already exist in results_<MARKET_CONFIG>/ (or results/ for
# three_firm). Train them first with:
#   python -m qlearning_collusion.run table1 --sessions 1000
#   python -m qlearning_collusion.run train rich_stochastic --sessions 1000
#   python -m qlearning_collusion.delta_sweep train --sessions 500
set -uo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-./venv/bin/python}
export MARKET_CONFIG=${MARKET_CONFIG:-three_firm_dist}
CELL=${CELL:-imperfect_stochastic}

step () {  # step <label> <command...>
  local label="$1"; shift
  echo ""
  echo "=== $label"
  if "$@"; then echo "    ok"; else echo "    FAILED: $label"; fi
}

step "0  network topology (all three structures)" \
    $PY -u -m qlearning_collusion.fig0_network
step "0b nodal demand curves / node-5 load pocket" \
    $PY -u -m qlearning_collusion.fig_demand
step "1-4 core figures (outputs, profits, limit strategy, deviation)" \
    $PY -u -m qlearning_collusion.run figures --name "$CELL"
step "5  is cheating deterred?" \
    $PY -u -c "from qlearning_collusion import figures; figures.fig5_deviation_value('$CELL')"
step "4b punishment, per deviating firm" \
    $PY -u -m qlearning_collusion.fig4b_punishment_split
step "6  deviation vs demand shock" \
    $PY -u -m qlearning_collusion.fig6_deviation_vs_shock
step "7  discount-factor sweep (Figure 4b per delta) + summary" \
    $PY -u -m qlearning_collusion.delta_sweep figures
step "8  LMP path / learning / punishment (cache)" \
    $PY -u -m qlearning_collusion.fig_lmp_learning --cache
step "8b LMP path / learning / punishment (draw)" \
    $PY -u -m qlearning_collusion.fig_lmp_learning
step "8c the four Table-I cells side by side" \
    $PY -u -c "from qlearning_collusion import fig_lmp_learning as F; F.fig_cells_compare()"
step "9  reaction curves (cache)" \
    $PY -u -m qlearning_collusion.fig_strategy --cache
step "9b reaction curves + why not grim trigger" \
    $PY -u -m qlearning_collusion.fig_strategy
step "R  transmission congestion report" \
    $PY -u -m qlearning_collusion.network_report "$CELL"
step "R2 Table I" \
    $PY -u -c "from qlearning_collusion.run import table1_text; print(table1_text())"

echo ""
echo "figures in qlearning_collusion/figures_${MARKET_CONFIG}/ (and figures_combined/)"
