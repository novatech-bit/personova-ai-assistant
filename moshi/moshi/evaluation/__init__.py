# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Full-Duplex-Bench evaluation integration for PersonaPlex/Moshi.

This package provides tooling to evaluate the model on the four key
interactive behaviours defined by Full-Duplex-Bench (v1/v1.5):

  1. **Pause Handling**      - Does the model wait appropriately during pauses?
  2. **Backchanneling**      - Does the model produce listener feedback at the right moments?
  3. **Smooth Turn-Taking**  - Can the model take and yield turns naturally?
  4. **User Interruption**   - Does the model stop speaking when the user interrupts?

See ``runner.py`` for the CLI-based batch evaluation harness and
``metrics.py`` for the metric computation utilities.
"""
