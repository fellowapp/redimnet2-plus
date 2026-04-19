#!/usr/bin/env bash
# Restore prior loader path when deactivating asv env.
if [ -n "${_ASV_OLD_LD_LIBRARY_PATH+x}" ]; then
  export LD_LIBRARY_PATH="${_ASV_OLD_LD_LIBRARY_PATH}"
  unset _ASV_OLD_LD_LIBRARY_PATH
else
  unset LD_LIBRARY_PATH
fi

if [ -n "${_ASV_OLD_PATH+x}" ]; then
  export PATH="${_ASV_OLD_PATH}"
  unset _ASV_OLD_PATH
fi