#!/bin/bash
# ============================================
# Retired gate, kept as a no-op placeholder.
#
# The corresponding backend has been sunset and its UT gate is obsolete,
# but an external CI job still invokes this script. It exits successfully
# so that gate stays green during the transition.
#
# TODO: delete this stub once the external CI job is removed.
# ============================================
echo "ci/run_ci_ms.sh: gate retired, nothing to do."
exit 0
