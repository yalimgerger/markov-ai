package com.markovai.server.ai.inference;

public class InferenceResult {
    private final int predictedDigit;
    private final double[] scores;
    private final double[] belief;
    private final int iterations;
    private final String topologyUsed;

    public InferenceResult(int predictedDigit, double[] scores, double[] belief, int iterations, String topologyUsed) {
        this.predictedDigit = predictedDigit;
        this.scores = scores;
        this.belief = belief;
        this.iterations = iterations;
        this.topologyUsed = topologyUsed;
    }

    public int getPredictedDigit() {
        return predictedDigit;
    }

    public double[] getScores() {
        return scores;
    }

    public double[] getBelief() {
        return belief;
    }

    public int getIterations() {
        return iterations;
    }

    public String getTopologyUsed() {
        return topologyUsed;
    }

    @Override
    public String toString() {
        return "InferenceResult{" +
                "predicted=" + predictedDigit +
                ", topScore="
                + (scores != null && predictedDigit >= 0 && predictedDigit < scores.length
                        ? String.format("%.4f", scores[predictedDigit])
                        : "N/A")
                +
                ", iterations=" + iterations +
                ", topology='" + topologyUsed + '\'' +
                '}';
    }
}
