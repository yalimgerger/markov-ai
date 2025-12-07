package com.markovai.server.ai;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.List;

public class RowColumnDigitClassifier {
    private static final Logger logger = LoggerFactory.getLogger(RowColumnDigitClassifier.class);
    private static final int NUM_DIGITS = 10;
    private static final int NUM_STATES = 16;

    private final DigitMarkovModel rowModel;
    private final DigitMarkovModel columnModel;
    private final DigitMarkovModel patchModel;

    private final MultiSequenceExtractor rowExtractor;
    private final MultiSequenceExtractor columnExtractor;
    private final SequenceExtractor patchExtractor;

    public RowColumnDigitClassifier() {
        this.rowExtractor = new RowPatchSequenceExtractor();
        this.columnExtractor = new ColumnPatchSequenceExtractor();
        this.patchExtractor = new PatchSequenceExtractor();

        this.rowModel = new DigitMarkovModel(NUM_STATES);
        this.columnModel = new DigitMarkovModel(NUM_STATES);
        this.patchModel = new DigitMarkovModel(NUM_STATES);
    }

    public DigitMarkovModel getRowModel() {
        return rowModel;
    }

    public DigitMarkovModel getColumnModel() {
        return columnModel;
    }

    public DigitMarkovModel getPatchModel() {
        return patchModel;
    }

    public MultiSequenceExtractor getRowExtractor() {
        return rowExtractor;
    }

    public MultiSequenceExtractor getColumnExtractor() {
        return columnExtractor;
    }

    public SequenceExtractor getPatchExtractor() {
        return patchExtractor;
    }

    public void train(List<DigitImage> trainingData) {
        logger.info("Starting Row-Column-Patch training with {} samples", trainingData.size());

        int[] digitCounts = new int[NUM_DIGITS];
        long totalRowSeqs = 0;
        long totalColSeqs = 0;

        for (DigitImage img : trainingData) {
            int d = img.label;
            if (d < 0 || d >= NUM_DIGITS)
                continue;
            digitCounts[d]++;

            int[][] binary = DigitMarkovModel.binarize(img.pixels, 128);

            List<int[]> rowSeqs = rowExtractor.extractSequences(binary);
            List<int[]> colSeqs = columnExtractor.extractSequences(binary);
            int[] patchSeq = patchExtractor.extractSequence(binary);

            rowModel.trainOnSequences(d, rowSeqs);
            columnModel.trainOnSequences(d, colSeqs);
            patchModel.trainOnSequences(d, java.util.Collections.singletonList(patchSeq));

            totalRowSeqs += rowSeqs.size();
            totalColSeqs += colSeqs.size();
        }

        for (int d = 0; d < NUM_DIGITS; d++) {
            logger.debug("Digit {}: {} samples processed", d, digitCounts[d]);
        }
        logger.info("Processed {} row sequences and {} column sequences.", totalRowSeqs, totalColSeqs);

        logger.info("Finalizing Row Model...");
        rowModel.finalizeProbabilities();
        logger.info("Finalizing Column Model...");
        columnModel.finalizeProbabilities();
        logger.info("Finalizing Patch Model...");
        patchModel.finalizeProbabilities();

        logger.info("Row-Column-Patch Training Complete.");
    }

    public ClassificationResult classifyWithScores(DigitImage img) {
        int[][] binary = DigitMarkovModel.binarize(img.pixels, 128);

        List<int[]> rowSeqs = rowExtractor.extractSequences(binary);
        List<int[]> colSeqs = columnExtractor.extractSequences(binary);

        if (logger.isDebugEnabled()) {
            logger.debug("Classifying image. Extracted {} row sequences, {} column sequences.", rowSeqs.size(),
                    colSeqs.size());
        }

        double[] logLikelihoods = new double[NUM_DIGITS];
        double[] surprises = new double[NUM_DIGITS];

        double bestTotalLogL = Double.NEGATIVE_INFINITY;
        double secondBestTotalLogL = Double.NEGATIVE_INFINITY;
        int bestDigit = -1;

        for (int d = 0; d < NUM_DIGITS; d++) {
            double rowLogL = rowModel.logLikelihoodForSequences(d, rowSeqs);
            double colLogL = columnModel.logLikelihoodForSequences(d, colSeqs);

            // Legacy behavior: Sum row + col only
            double totalLogL = rowLogL + colLogL;

            logLikelihoods[d] = totalLogL;
            surprises[d] = -totalLogL;

            if (logger.isTraceEnabled()) {
                logger.trace("Digit {}: RowLogL={:.2f}, ColLogL={:.2f}, Total={:.2f}", d, rowLogL, colLogL, totalLogL);
            }

            if (totalLogL > bestTotalLogL) {
                secondBestTotalLogL = bestTotalLogL;
                bestTotalLogL = totalLogL;
                bestDigit = d;
            } else if (totalLogL > secondBestTotalLogL) {
                secondBestTotalLogL = totalLogL;
            }
        }

        logger.info("Classified as digit: {}. MaxLogL={:.2f}, DiffToSecond={:.2f}", bestDigit, bestTotalLogL,
                (bestTotalLogL - secondBestTotalLogL));

        return new ClassificationResult(bestDigit, logLikelihoods, surprises);
    }

    public int classify(DigitImage img) {
        return classifyWithScores(img).getPredictedDigit();
    }

    public void evaluateAccuracy(List<DigitImage> testData) {
        logger.info("Evaluating accuracy on {} test images...", testData.size());
        int correct = 0;
        int total = 0;

        for (DigitImage img : testData) {
            int predicted = classify(img);
            if (predicted == img.label) {
                correct++;
            }
            total++;
            if (total % 1000 == 0) {
                logger.info("Evaluated {}/{}...", total, testData.size());
            }
        }

        double accuracy = (double) correct / total;
        logger.info("Evaluation complete. Accuracy: {:.4f} ({}/{})", accuracy, correct, total);
    }
}
