package com.markovai.server.ai;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.List;

/**
 * Manages Markov models for all 10 digits and handles classification.
 */
public class DigitMarkovModel {
    private static final Logger logger = LoggerFactory.getLogger(DigitMarkovModel.class);
    private static final int NUM_DIGITS = 10;
    private static final int IMAGE_SIZE = 28;

    private final int numStates;
    private final SequenceExtractor extractor;

    // counts and probabilities for each digit 0..9
    // [digit][state]
    private final long[][] initialCounts;
    // [digit][prevState][nextState]
    private final long[][][] transitionCounts;

    // [digit][state]
    private final double[][] initialProbs;
    // [digit][prevState][nextState]
    private final double[][][] transitionProbs;

    public DigitMarkovModel(int numStates, SequenceExtractor extractor) {
        this.numStates = numStates;
        this.extractor = extractor;

        this.initialCounts = new long[NUM_DIGITS][numStates];
        this.transitionCounts = new long[NUM_DIGITS][numStates][numStates];
        this.initialProbs = new double[NUM_DIGITS][numStates];
        this.transitionProbs = new double[NUM_DIGITS][numStates][numStates];

        logger.info("Initializing DigitMarkovModel with {} states. Using extractor: {}", numStates,
                extractor.getClass().getSimpleName());
    }

    /**
     * Helper to binarize a 28x28 grayscale image.
     * 
     * @param pixels    28x28 int array with values 0-255
     * @param threshold values >= threshold become 1, else 0
     * @return 28x28 int arrays with values 0 or 1
     */
    public static int[][] binarize(int[][] pixels, int threshold) {
        int[][] binary = new int[IMAGE_SIZE][IMAGE_SIZE];
        for (int r = 0; r < IMAGE_SIZE; r++) {
            for (int c = 0; c < IMAGE_SIZE; c++) {
                binary[r][c] = pixels[r][c] >= threshold ? 1 : 0;
            }
        }
        return binary;
    }

    public void train(List<DigitImage> trainingData) {
        logger.info("Starting training with {} samples", trainingData.size());
        long startTime = System.currentTimeMillis();

        // Initialize counts to 0 (Java does this by default, but for clarity...)
        // Skipped explicit zeroing as arrays are fresh.

        int[] digitCounts = new int[NUM_DIGITS];

        for (DigitImage img : trainingData) {
            int[][] binary = binarize(img.pixels, 128);
            int[] seq = extractor.extractSequence(binary);

            // Log for first few samples if trace enabled, checking expected values
            if (logger.isTraceEnabled() && digitCounts[img.label] == 0) {
                logger.trace("Extracted sequence length: {}", seq.length);
                if (seq.length > 5) {
                    logger.trace("First 5 states: {}, {}, {}, {}, {}", seq[0], seq[1], seq[2], seq[3], seq[4]);
                }
            }

            int d = img.label;

            if (d < 0 || d >= NUM_DIGITS)
                continue;
            digitCounts[d]++;

            // Update initial state count
            if (seq[0] >= 0 && seq[0] < numStates) {
                initialCounts[d][seq[0]]++;
            }

            // Update transitions
            for (int t = 1; t < seq.length; t++) {
                int prev = seq[t - 1];
                int cur = seq[t];
                if (prev >= 0 && prev < numStates && cur >= 0 && cur < numStates) {
                    transitionCounts[d][prev][cur]++;
                }
            }
        }

        for (int d = 0; d < NUM_DIGITS; d++) {
            logger.debug("Digit {}: {} training samples", d, digitCounts[d]);
        }

        finalizeProbabilities();

        long duration = System.currentTimeMillis() - startTime;
        logger.info("Training complete in {} ms", duration);
    }

    private void finalizeProbabilities() {
        for (int d = 0; d < NUM_DIGITS; d++) {
            // Initial probabilities: Calculate total count + smoothing mass
            long totalInit = numStates; // Add '1' for each state (Laplace smoothing)
            for (int s = 0; s < numStates; s++) {
                totalInit += initialCounts[d][s];
            }

            for (int s = 0; s < numStates; s++) {
                initialProbs[d][s] = (double) (initialCounts[d][s] + 1) / totalInit;
            }

            // Transition probabilities
            for (int prev = 0; prev < numStates; prev++) {
                long totalTrans = numStates; // Add '1' for each next state
                for (int next = 0; next < numStates; next++) {
                    totalTrans += transitionCounts[d][prev][next];
                }

                for (int next = 0; next < numStates; next++) {
                    transitionProbs[d][prev][next] = (double) (transitionCounts[d][prev][next] + 1) / totalTrans;
                }

                if (logger.isTraceEnabled() && numStates <= 4) { // Only log full tables if small
                    for (int next = 0; next < numStates; next++) {
                        logger.trace("Digit {} transition {}->{}: count={}, prob={}", d, prev, next,
                                transitionCounts[d][prev][next], transitionProbs[d][prev][next]);
                    }
                }
            }
        }
        logger.debug("Probabilities finalized for {} states.", numStates);
    }

    public double logLikelihood(int digit, int[] seq) {
        // Removed hard seq length check as it varies by extractor

        if (seq[0] < 0 || seq[0] >= numStates)
            return Double.NEGATIVE_INFINITY; // Should not happen

        double p0 = initialProbs[digit][seq[0]];
        double logL = Math.log(p0);

        for (int t = 1; t < seq.length; t++) {
            int prev = seq[t - 1];
            int cur = seq[t];
            if (prev >= 0 && prev < numStates && cur >= 0 && cur < numStates) {
                double p = transitionProbs[digit][prev][cur];
                logL += Math.log(p);
            }
        }
        return logL;
    }

    public double surprise(int digit, int[] seq) {
        return -logLikelihood(digit, seq);
    }

    public int classify(DigitImage img) {
        return classifyWithScores(img).getPredictedDigit();
    }

    public ClassificationResult classifyWithScores(DigitImage img) {
        int[][] binary = binarize(img.pixels, 128);
        int[] seq = extractor.extractSequence(binary);

        // Debug logging for extraction
        if (logger.isDebugEnabled()) {
            logger.debug("Using extractor: {}", extractor.getClass().getSimpleName());
            logger.debug("Extracted sequence length: {}", seq.length);
            if (seq.length >= 5) {
                logger.debug("First 5 states of input: {}, {}, {}, {}, {}", seq[0], seq[1], seq[2], seq[3], seq[4]);
            }
        }

        double[] logLikelihoods = new double[NUM_DIGITS];
        double[] surprises = new double[NUM_DIGITS];

        double bestLogL = Double.NEGATIVE_INFINITY;
        int bestDigit = -1;

        // Log input identification hash/snippet
        if (logger.isTraceEnabled()) {
            // Print full sequence only in trace
            logger.trace("Classifying sequence: " + java.util.Arrays.toString(seq));
        }

        for (int d = 0; d < NUM_DIGITS; d++) {
            double logL = logLikelihood(d, seq);
            logLikelihoods[d] = logL;
            surprises[d] = -logL;

            logger.debug("Digit {} log-likelihood = {}", d, logL);

            if (logL > bestLogL) {
                bestLogL = logL;
                bestDigit = d;
            }
        }

        logger.info("Classified as digit: {}", bestDigit);

        return new ClassificationResult(bestDigit, logLikelihoods, surprises);
    }

    public double evaluateAccuracy(List<DigitImage> testData) {
        int correct = 0;
        int total = testData.size();

        logger.info("Evaluating accuracy on {} samples...", total);

        for (DigitImage img : testData) {
            int predicted = classify(img);
            if (predicted == img.label) {
                correct++;
            } else {
                logger.debug("Misclassified label {} as {}", img.label, predicted);
            }
        }

        double accuracy = (double) correct / total;
        logger.info("Evaluation complete. Accuracy: {} ({}/{})", accuracy, correct, total);
        return accuracy;
    }
}
