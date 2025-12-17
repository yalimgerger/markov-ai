package com.markovai.server.ai.inference;

import java.util.Arrays;

public class MathUtil {

    /**
     * Computes the softmax of an array of scores.
     * Uses the "max trick" for numerical stability:
     * softmax(x_i) = exp(x_i - max(x)) / sum(exp(x_j - max(x)))
     */
    public static double[] softmax(double[] scores, double temperature) {
        if (temperature <= 0) {
            throw new IllegalArgumentException("Temperature must be positive");
        }

        double max = Double.NEGATIVE_INFINITY;
        for (double s : scores) {
            if (s > max)
                max = s;
        }

        double[] probs = new double[scores.length];
        double sum = 0.0;
        for (int i = 0; i < scores.length; i++) {
            double val = Math.exp((scores[i] - max) / temperature);
            probs[i] = val;
            sum += val;
        }

        for (int i = 0; i < probs.length; i++) {
            probs[i] /= sum;
        }
        return probs;
    }

    /**
     * Computes the entropy of a probability distribution (base e or base 2).
     * Using base e (natural log) for typical information theory entropy.
     * H(p) = -sum(p_i * log(p_i))
     */
    public static double entropy(double[] probs) {
        double h = 0.0;
        for (double p : probs) {
            if (p > 1e-12) {
                h -= p * Math.log(p);
            }
        }
        return h;
    }

    /**
     * Returns the index of the maximum value in the array.
     */
    public static int argmax(double[] x) {
        int bestIdx = -1;
        double bestVal = Double.NEGATIVE_INFINITY;
        for (int i = 0; i < x.length; i++) {
            if (x[i] > bestVal) {
                bestVal = x[i];
                bestIdx = i;
            }
        }
        return bestIdx;
    }

    /**
     * Computes the maximum absolute difference between two arrays.
     */
    public static double maxAbsDelta(double[] a, double[] b) {
        if (a.length != b.length) {
            throw new IllegalArgumentException("Arrays must have same length");
        }
        double maxDelta = 0.0;
        for (int i = 0; i < a.length; i++) {
            double delta = Math.abs(a[i] - b[i]);
            if (delta > maxDelta) {
                maxDelta = delta;
            }
        }
        return maxDelta;
    }
}
