package com.markovai.server.ai.inference;

public class NetworkInferenceResult extends InferenceResult {
    private final double initialEntropy;
    private final double finalEntropy;
    private final double finalMaxDelta;
    private final boolean oscillationDetected;

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
        super(predictedDigit, scores, belief, iterations, topologyUsed);
        this.initialEntropy = initialEntropy;
        this.finalEntropy = finalEntropy;
        this.finalMaxDelta = finalMaxDelta;
        this.oscillationDetected = oscillationDetected;
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
}
