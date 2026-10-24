#!/usr/bin/env bash
# Watch the pod's tmux session "sinmulu" live from this computer (read-only use; closing this window or Ctrl+C
# only disconnects the viewer, the runs on the pod continue). The RunPod SSH proxy accepts no remote command,
# so "tmux attach" is typed into the session through stdin.
POD="${POD:-wmwc57eij52665-64412316@ssh.runpod.io}"
(printf 'tmux attach -t sinmulu\n'; cat) | ssh -tt -o LogLevel=ERROR -i ~/.ssh/id_ed25519 "$POD"
