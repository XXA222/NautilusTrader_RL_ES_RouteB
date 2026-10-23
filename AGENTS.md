# Canonical workspace policy

The one and only writable local checkout for this project is:

`D:\Program Files\NautilusTrader_RL_ES_RouteB`

Before editing, generating files, installing an environment, or running a command that can
write to the checkout, resolve the repository root and run:

```powershell
python qa/assert_canonical_workspace.py
```

If the check fails, stop. Do not write to that checkout, do not create another RouteB clone,
and do not silently fall back to a similarly named directory. Reopen or move the work to the
canonical path first.

The RouteA repositories `D:\Program Files\NautilusTrader_RL_ES` and
`D:\Program Files\NautilusTrader_RL_ES_IL_Route_A` are separate projects. Never write RouteB
changes into them and never merge RouteA into RouteB merely to work around a path problem.

Hosted CI runners may check out the repository elsewhere for isolated build and test work.
That exception does not authorize local development or persistent outputs outside the canonical
directory. Temporary and generated outputs must remain under the canonical project root or the
operating system's temporary directory and must not create another project checkout.
