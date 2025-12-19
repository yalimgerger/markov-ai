package com.markovai.server.ai.learning;

import com.markovai.server.ai.hierarchy.FactorGraphBuilder;
import com.markovai.server.ai.inference.InferenceResult;
import com.markovai.server.ai.inference.NetworkInferenceResult;

public class PayoffCalculator {

    public static double computePayoffScale(
            InferenceResult result,
            int trueLabel,
            FactorGraphBuilder.PayoffConfig config) {

        if (config == null || config.enabled == null || !config.enabled) {
            return 1.0;
        }

        // Only applicable to network results which have belief distributions
        if (!(result instanceof NetworkInferenceResult)) {
            return 1.0;
        }

        NetworkInferenceResult netResult = (NetworkInferenceResult) result;
        double[] belief = netResult.getBelief();

        // 1. Identify Top 1 and Top 2
        int top1 = -1;
        int top2 = -1;
        double max1 = -1.0;
        double max2 = -1.0;

        for (int i = 0; i < belief.length; i++) {
            double p = belief[i];
            if (p > max1) {
                max2 = max1;
                top2 = top1;
                max1 = p;
                top1 = i;
            } else if (p > max2) {
                max2 = p;
                top2 = i;
            }
        }

        double confGap = max1 - max2;
        boolean wasCorrect = (top1 == trueLabel);

        // 2. Check Convergence
        // We use iterations rule primarily, and stopEpsilon if available (implied by
        // iterations < max)
        // If maxItersForConverged is set, we check against it.
        boolean converged = true;
        if (config.requireConvergence != null && config.requireConvergence) {
            int maxIters = config.maxItersForConverged != null ? config.maxItersForConverged : Integer.MAX_VALUE;
            // If iterations used >= maxItersForConverged, we assume it didn't converge
            // "fast enough" or "well enough"
            // for a high payoff, or it might be oscillating.
            if (netResult.getIterations() >= maxIters) {
                converged = false;
            }

            // Optionally check oscillation bit if we wanted to be stricter, but configured
            // rule is just iters
        }

        if (!converged) {
            return 0.0;
        }

        // 3. Compute Base Payoff
        double scale = 0.0;
        double confStrong = config.confStrong != null ? config.confStrong : 0.25;
        double confWeak = config.confWeak != null ? config.confWeak : 0.10;
        double scaleStrong = config.scaleStrong != null ? config.scaleStrong : 1.0;
        double scaleWeakCorrect = config.scaleWeakCorrect != null ? config.scaleWeakCorrect : 0.30;
        double scaleWeakIncorrect = config.scaleWeakIncorrect != null ? config.scaleWeakIncorrect : 0.20;

        if (confGap >= confStrong) {
            scale = scaleStrong;
        } else if (confGap >= confWeak) {
            scale = wasCorrect ? scaleWeakCorrect : scaleWeakIncorrect;
        } else {
            scale = 0.0;
        }

        // 4. Correctness Gating
        boolean applyToCorrect = config.applyToCorrect != null ? config.applyToCorrect : true;
        boolean applyToIncorrect = config.applyToIncorrect != null ? config.applyToIncorrect : true;

        if (wasCorrect && !applyToCorrect) {
            return 0.0;
        }
        if (!wasCorrect && !applyToIncorrect) {
            return 0.0;
        }

        return scale;
    }
}
