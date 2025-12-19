package com.markovai.server.ai.inference;

import java.util.List;

public class NetworkInferenceResult extends InferenceResult {
    private final double initialEntropy;
    private final double finalEntropy;
    private final double finalMaxDelta;
    private final boolean oscillationDetected;
    private final List<double[]> beliefTrajectory;

    public NetworkInferenceResult(
            int predictedDigit,
            double[] scores,
            double[] belief,
            int iterations,
            String topologyUsed,
            double initialEntropy,
            double finalEntropy,
            double finalMaxDelta,
            boolean oscillationDetected,
            List<double[]> beliefTrajectory) {
        super(predictedDigit, scores, belief, iterations, topologyUsed);
        this.initialEntropy = initialEntropy;
        this.finalEntropy = finalEntropy;
        this.finalMaxDelta = finalMaxDelta;
        this.oscillationDetected = oscillationDetected;
        this.beliefTrajectory = beliefTrajectory;
    }

    public NetworkInferenceResult(
            int predictedDigit,
            double[] scores,
            double[] belief,
            int iterations,
            String topologyUsed,
            double initialEntropy,
            double finalEntropy,
            double finalMaxDelta,
            boolean oscillationDetected) {
        this(predictedDigit, scores, belief, iterations, topologyUsed, initialEntropy, finalEntropy, finalMaxDelta,
                oscillationDetected, null);
    }

    public double getInitialEntropy() {
        return initialEntropy;
    }

    public double getFinalEntropy() {
        return finalEntropy;
    }

    public double getFinalMaxDelta() {
        return finalMaxDelta;
    }

    public boolean isOscillationDetected() {
        return oscillationDetected;
    }

    public List<double[]> getBeliefTrajectory() {
        return beliefTrajectory;
    }
}
