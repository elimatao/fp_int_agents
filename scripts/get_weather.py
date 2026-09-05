#!/usr/bin/env python3
"""Example tool script. Reads JSON args from stdin, writes result to stdout."""

import json
import sys

args = json.loads(sys.stdin.read())
location = args.get("location", "unknown")
units = args.get("units", "celsius")

# Stub response — replace with a real weather API call.
temp = 22 if units == "celsius" else 72
print(f"Weather in {location}: {temp}°{'C' if units == 'celsius' else 'F'}, sunny.")
