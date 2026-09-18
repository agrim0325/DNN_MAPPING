# Paper-reproduction next steps

This plan begins at local commit `945b14f`. Work is being prepared on local
branch `codex/paper-next-steps` and must not be pushed until the junior's remote
changes have been reviewed and integrated.

## Execution order

1. **Establish learning evidence.** Run five matched seeds for MLP and CNN with
   1,000 complete-placement evaluations per method. Analyze deterministic and
   noisy DDPG costs separately. Do not extend to 10,000 epochs unless the
   deterministic policy is still improving near epoch 1,000.
2. **Improve credit assignment experimentally.** Retain paper-style sparse
   terminal reward as the default. Add potential-based shaping as an opt-in
   condition whose discounted shaping return telescopes to zero, preserving
   the original objective for a fixed episode horizon.
3. **Validate architecture claims.** Compare the spatial CNN implementation
   with the actor/critic diagram and dimensions in the paper. Record every
   unspecified layer or hyperparameter as an assumption.
4. **Reconstruct the timing model.** Specify block size, pipeline stages,
   transformation work, communication startup, bandwidth sharing, contention,
   buffering, stalls and compute/communication overlap. Add small hand-worked
   examples before using the model in long experiments.
5. **Improve partition fidelity.** Add compute-aware partitions, weight and
   activation-buffer capacity checks, merge arithmetic, and any required
   CONV/FC region constraints.
6. **Run paper-scale baselines.** Increase RS and SA toward the paper's search
   budget only after objective validation. Report placement evaluations,
   runtime, seed-level results and uncertainty.
7. **Reproduce, then improve.** Freeze a paper-mode configuration. Evaluate
   dense reward, discrete/masked actions, graph encoders and parallel
   environments as separately labeled improvements.

## Evidence required for each experiment

- Git commit and dirty/clean status.
- Full configuration, Torch/CUDA versions and GPU name.
- Seed-level reports and checkpoints.
- Noisy current cost, deterministic-policy cost and best-ever cost.
- Actor/critic losses, collision repairs and exploration scale.
- Mean, sample standard deviation, minimum and maximum over at least five seeds.
- Matched placement-evaluation budgets for DDPG, random search and SA.
