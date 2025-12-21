package com.markovai.server.ai;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.List;

/**
 * A unigram model for Gradient Orientation symbols.
 * 
 * Image -> 7x7 grid of gradient symbols (0..numBins).
 * Training: Count symbol frequencies per digit.
 * Scoring: Sum of log probabilities of symbols.
 */
public class DigitGradientUnigramModel {
    private static final Logger logger = LoggerFactory.getLogger(DigitGradientUnigramModel.class);

    private final int numBins; // e.g. 8
    private final int flatBin; // e.g. 8 (total symbols = numBins + 1)
    private final int encodedSize; // numBins + 1

    private final int blockSize;
    private final double magThreshold;
    private final double smoothingAlpha;

    // [digit][symbol] -> count
    private final long[][] counts;
    // [digit][symbol] -> logProb
    private final double[][] logProbs;

    public DigitGradientUnigramModel(int blockSize, int numBins, int flatBin, double magThreshold,
            double smoothingAlpha) {
        this.blockSize = blockSize;
        this.numBins = numBins;
        this.flatBin = flatBin;
        this.encodedSize = numBins + 1;
        this.magThreshold = magThreshold;
        this.smoothingAlpha = smoothingAlpha;

        this.counts = new long[10][encodedSize];
        this.logProbs = new double[10][encodedSize];
    }

    public int getNumBins() {
        return numBins;
    }

    public int getFlatBin() {
        return flatBin;
    }

    public int getBlockSize() {
        return blockSize;
    }

    public double getMagThreshold() {
        return magThreshold;
    }

    public void train(List<DigitImage> trainingData) {
        logger.info("Training GradientUnigramModel on {} images. BlockSize={}, Bins={}, MagTh={}", trainingData.size(),
                blockSize, numBins, magThreshold);

        long start = System.currentTimeMillis();

        // Reset counts
        for (int d = 0; d < 10; d++) {
            for (int s = 0; s < encodedSize; s++) {
                counts[d][s] = 0;
            }
        }

        for (DigitImage img : trainingData) {
            int d = img.label;
            if (d < 0 || d > 9)
                continue;

            int[] symbols = extractSymbols(img.pixels);
            for (int s : symbols) {
                if (s >= 0 && s < encodedSize) {
                    counts[d][s]++;
                }
            }
        }

        // Finalize probabilities
        for (int d = 0; d < 10; d++) {
            long total = 0;
            for (int s = 0; s < encodedSize; s++)
                total += counts[d][s];

            double denom = total + (encodedSize * smoothingAlpha);

            for (int s = 0; s < encodedSize; s++) {
                double num = counts[d][s] + smoothingAlpha;
                logProbs[d][s] = Math.log(num / denom);
            }
        }

        logger.info("Gradient Model Training complete in {} ms", (System.currentTimeMillis() - start));
        logStats();
    }

    private void logStats() {
        for (int d = 0; d < 10; d++) {
            long total = 0;
            long flats = counts[d][flatBin];
            for (int s = 0; s < encodedSize; s++)
                total += counts[d][s];
            double flatPct = (double) flats / total * 100.0;
            logger.info("Digit {}: Total Symbols={}, FlatBin%={}%", d, total, String.format("%.2f", flatPct));
        }
    }

    // Scoring
    public double score(int digit, int[] symbols) {
        double logL = 0.0;
        for (int s : symbols) {
            if (s >= 0 && s < encodedSize) {
                logL += logProbs[digit][s];
            }
        }
        return logL;
    }

    public int[] extractSymbols(int[][] pixels) {
        int h = pixels.length;
        int w = pixels[0].length;

        int rows = h / blockSize;
        int cols = w / blockSize;

        int[] symbols = new int[rows * cols];
        int idx = 0;

        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                symbols[idx++] = computeBlockSymbol(pixels, r * blockSize, c * blockSize);
            }
        }
        return symbols;
    }

    private int computeBlockSymbol(int[][] pixels, int r0, int c0) {
        double sumDx = 0.0;
        double sumDy = 0.0;

        // Simple finite difference within block
        // Be careful with image boundaries

        int rEnd = r0 + blockSize;
        int cEnd = c0 + blockSize;

        int h = pixels.length;
        int w = pixels[0].length;

        for (int r = r0; r < rEnd; r++) {
            for (int c = c0; c < cEnd; c++) {
                // Sobel-ish or simple difference
                // dx = I(r, c+1) - I(r, c-1)
                // dy = I(r+1, c) - I(r-1, c)

                int valPlusX = (c + 1 < w) ? pixels[r][c + 1] : pixels[r][c];
                int valMinusX = (c - 1 >= 0) ? pixels[r][c - 1] : pixels[r][c];

                int valPlusY = (r + 1 < h) ? pixels[r + 1][c] : pixels[r][c];
                int valMinusY = (r - 1 >= 0) ? pixels[r - 1][c] : pixels[r][c];

                sumDx += (valPlusX - valMinusX);
                sumDy += (valPlusY - valMinusY);
            }
        }

        double mag = Math.sqrt(sumDx * sumDx + sumDy * sumDy);

        if (mag < magThreshold) {
            return flatBin;
        }

        // atan2 returns -pi to pi
        double angle = Math.atan2(sumDy, sumDx); // in radians
        if (angle < 0)
            angle += 2 * Math.PI; // 0 to 2pi

        // e.g. 8 bins => 0..7
        // bin size = 2pi / 8
        double binSize = (2 * Math.PI) / numBins;
        int bin = (int) (angle / binSize);
        if (bin >= numBins)
            bin = numBins - 1;

        return bin;
    }
}
